# selection_manager/management/commands/prepare_csi300_features.py

import logging
import pickle
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from tqdm import tqdm
import pandas_ta as ta

from common.models import IndexQuotesCsi300

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '[Gen2] 构建多特征3D张量，并应用Triple-Barrier三分类标签。'

    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    FEATURES_FILE = MODELS_DIR / 'csi300_features_gen2.pkl'
    LOOKBACK_WINDOW = 60
    
    # --- [新] 特征和标签配置 ---
    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
    DERIVED_FEATURE_COLS = ['log_return', 'volatility_20d', 'bias_20d', 'bias_60d']
    ALL_FEATURE_COLS = FEATURE_COLS + DERIVED_FEATURE_COLS
    N_FEATURES = len(ALL_FEATURE_COLS)
    
    # Triple-Barrier Method 配置
    TBM_LOOKFORWARD = 10  # 向前看10天
    TBM_PROFIT_TAKE_MULT = 1.5 # 止盈倍数
    TBM_STOP_LOSS_MULT = 1.0 # 止损倍数

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加衍生特征"""
        df['log_return'] = np.log(df['close']).diff().fillna(0)
        df['volatility_20d'] = df['log_return'].rolling(self.LOOKBACK_WINDOW).std().fillna(0)
        
        ma20 = df['close'].rolling(20).mean()
        ma60 = df['close'].rolling(60).mean()
        df['bias_20d'] = (df['close'] / ma20 - 1).fillna(0)
        df['bias_60d'] = (df['close'] / ma60 - 1).fillna(0)
        
        return df

    def _get_triple_barrier_labels(self, prices: pd.Series, atr: pd.Series) -> pd.Series:
        """计算三分类标签"""
        labels = pd.Series(0, index=prices.index) # 默认标签为0(盘整)
        
        for i in tqdm(range(len(prices) - self.TBM_LOOKFORWARD), desc="Applying Triple-Barrier"):
            current_price = prices.iloc[i]
            current_atr = atr.iloc[i]
            
            if current_atr <= 0: continue

            upper_barrier = current_price + self.TBM_PROFIT_TAKE_MULT * current_atr
            lower_barrier = current_price - self.TBM_STOP_LOSS_MULT * current_atr
            
            future_prices = prices.iloc[i+1 : i+1+self.TBM_LOOKFORWARD]
            
            # 检查是否触及上轨
            hit_upper = future_prices[future_prices >= upper_barrier]
            # 检查是否触及下轨
            hit_lower = future_prices[future_prices <= lower_barrier]
            
            if not hit_upper.empty and not hit_lower.empty:
                # 如果都触及，看哪个先到
                if hit_upper.index[0] <= hit_lower.index[0]:
                    labels.iloc[i] = 1 # 上涨
                else:
                    labels.iloc[i] = 2 # 下跌
            elif not hit_upper.empty:
                labels.iloc[i] = 1 # 上涨
            elif not hit_lower.empty:
                labels.iloc[i] = 2 # 下跌
        
        return labels

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== [Gen2] 开始准备机器学习特征 ====="))
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. 加载数据
        quotes_qs = IndexQuotesCsi300.objects.all().order_by('trade_date').values('trade_date', *self.FEATURE_COLS)
        df = pd.DataFrame.from_records(quotes_qs)
        df.set_index('trade_date', inplace=True)
        for col in self.FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # 2. 添加衍生特征
        self.stdout.write("步骤 1/4: 添加衍生特征...")
        df = self._add_derived_features(df)
        
        # 计算 ATR 用于标签生成
        df['atr'] = df.ta.atr(length=14) 
        df.dropna(inplace=True)

        # 3. 生成三分类标签
        self.stdout.write("步骤 2/4: 应用 Triple-Barrier 方法生成标签...")
        labels = self._get_triple_barrier_labels(df['close'], df['atr'])
        df['label'] = labels

        # 4. 构建3D特征张量 & 对齐数据
        self.stdout.write("步骤 3/4: 构建3D特征张量...")
        X_data, y_data, index_data = [], [], []

        for i in tqdm(range(self.LOOKBACK_WINDOW, len(df))):
            feature_window = df.iloc[i - self.LOOKBACK_WINDOW : i][self.ALL_FEATURE_COLS]
            label = df.iloc[i-1]['label'] # 用T-1日的特征，预测从T日开始的未来走势
            
            if label == 0: continue # 过滤掉盘整样本，让模型专注于有意义的行情
            
            # 使用 Z-Score 归一化窗口内数据
            mean = feature_window.mean()
            std = feature_window.std()
            std[std == 0] = 1e-9 # 防止除以零
            normalized_window = (feature_window - mean) / std

            if normalized_window.isnull().values.any(): continue
            
            X_data.append(normalized_window.values)
            y_data.append(label)
            index_data.append(feature_window.index[-1])

        X = np.array(X_data)
        y = np.array(y_data)
        
        # 将标签从 (1, 2) 映射到 (0, 1)
        y[y == 1] = 0 # 上涨
        y[y == 2] = 1 # 下跌

        self.stdout.write(f"过滤盘整样本后，剩余样本数: {len(X)}")
        self.stdout.write(f"上涨样本数 (标签0): {(y == 0).sum()}, 下跌样本数 (标签1): {(y == 1).sum()}")
        
        # 5. 保存
        self.stdout.write("步骤 4/4: 保存特征数据...")
        feature_data = {'X': X, 'y': y, 'index': index_data}
        with open(self.FEATURES_FILE, 'wb') as f:
            pickle.dump(feature_data, f)
            
        self.stdout.write(self.style.SUCCESS(f"===== [Gen2] 特征准备成功！文件保存在: {self.FEATURES_FILE} ====="))
