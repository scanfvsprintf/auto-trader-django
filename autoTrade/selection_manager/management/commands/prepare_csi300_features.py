# selection_manager/management/commands/prepare_csi300_features.py

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from tqdm import tqdm

from common.models import IndexQuotesCsi300
from decimal import Decimal
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '从数据库读取沪深300数据，提取特征向量并保存至文件。'

    # ... (其他类变量不变) ...
    WINDOW_SIZE = 60
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    FEATURES_FILE = MODELS_DIR / 'csi300_features.pkl'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== 开始准备沪深300指数的机器学习特征 ====="))

        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.stdout.write("步骤 1/4: 从数据库加载沪深300行情数据...")
        try:
            quotes = IndexQuotesCsi300.objects.all().order_by('trade_date').values()
            if not quotes:
                self.stderr.write(self.style.ERROR("错误：数据库中没有沪深300指数数据。"))
                return
            df = pd.DataFrame.from_records(quotes)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            self.stdout.write(f"成功加载 {len(df)} 条数据。")

            ## ======================== [核心修复] ========================
            # 使用更健壮的方式检查和转换 Decimal 类型
            # ==========================================================
            for col in df.columns:
                # 检查该列是否有任何 Decimal 类型的实例
                # 我们只检查第一个非空值，这样效率更高
                first_valid_element = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid_element, Decimal):
                    self.stdout.write(f"检测到列 '{col}' 为 Decimal 类型，正在转换为 float...")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.stdout.write("数据类型检查和转换完成。")

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"加载数据或类型转换失败: {e}"))
            return

        self.stdout.write("步骤 2/4: 计算技术指标...")
        
        # 现在所有计算都将在 float 类型上进行，不会再有类型错误
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        # 传递原始价格列给ta函数，确保它们是正确的float类型
        df.ta.bbands(close=df['close'], length=20, append=True)
        df.ta.atr(high=df['high'], low=df['low'], close=df['close'], append=True)
        df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], append=True)
        df.ta.willr(high=df['high'], low=df['low'], close=df['close'], append=True)
        
        df.dropna(inplace=True)
        
        # 整理特征列
        # 明确指定需要作为特征的列，更健壮
        all_cols = set(df.columns)
        price_cols = {'open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_change', 'change_amount', 'turnover_rate'}
        feature_cols = list(all_cols - price_cols- {'target'})
        feature_df = df[feature_cols]

        self.stdout.write(f"技术指标计算完成，剩余 {len(feature_df)} 条有效数据。")

        # 3. 构建特征向量(X)和目标(y)
        self.stdout.write("步骤 3/4: 构建特征向量 X 和目标 y...")
        
        df['target'] = df['close'].pct_change().shift(-1)
        df.dropna(subset=['target'], inplace=True)
        
        # 确保特征和目标在相同的索引上对齐
        common_index = df.index.intersection(feature_df.index)
        df = df.loc[common_index]
        feature_df = feature_df.loc[common_index]

        X, y = [], []
        # 使用tqdm创建进度条
        for i in tqdm(range(self.WINDOW_SIZE, len(feature_df)), desc="构建样本"):
            feature_window = feature_df.iloc[i - self.WINDOW_SIZE : i].values
            X.append(feature_window.flatten())
            y.append(df['target'].iloc[i - 1]) # 修正此处，应为 i-1

        if not X:
            self.stderr.write(self.style.ERROR("未能构建任何特征样本，可能是数据量不足。"))
            return

        self.stdout.write(f"成功构建 {len(X)} 个训练样本。")

        # 4. 保存特征和目标到文件 (代码不变)
        self.stdout.write(f"步骤 4/4: 保存特征数据到 {self.FEATURES_FILE}...")
        
        feature_data = {
            'X': np.array(X),
            'y': np.array(y),
            'feature_names': feature_df.columns.tolist(),
            'window_size': self.WINDOW_SIZE
        }

        try:
            with open(self.FEATURES_FILE, 'wb') as f:
                pickle.dump(feature_data, f)
            self.stdout.write(self.style.SUCCESS("===== 特征准备成功！ ====="))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"保存特征文件失败: {e}"))
