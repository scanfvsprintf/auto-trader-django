# ==============================================================================
# 文件 1/4: selection_manager/management/commands/prepare_stock_features.py
# 描述: 为个股评分模型生成特征和标签数据集。
# ==============================================================================
import logging
import pickle
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
from django.core.management.base import BaseCommand
from django.conf import settings
from tqdm import tqdm

from common.models import DailyQuotes, DailyFactorValues
from selection_manager.service.m_value_service import FactorCalculator # 复用M值服务中的因子计算器
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE

logger = logging.getLogger(__name__)

# --- [配置区] ---
# 通过修改此处的配置，来控制数据集的生成方式
LABEL_CONFIG = {
    'mode': 'sharpe',  # 'return' (未来收益率) 或 'sharpe' (未来夏普比率)
    'lookforward_days': 10, # 标签向前看的天数 (N)
    'risk_free_rate_annual': 0.02, # 年化无风险利率，仅在 'sharpe' 模式下使用
    'tanh_scaling_factor': 1.0, # tanh缩放因子，仅在 'sharpe' 模式下使用
}

# 因子计算所需的最大回溯期，应大于所有因子中最大的lookback period
# 例如，neg_dev(60), max_dd(60)等，给足100天buffer
FACTOR_LOOKBACK_BUFFER = 100

class Command(BaseCommand):
    help = '为个股评分模型生成特征和标签数据集 (X, y)。'

    # --- 路径配置 ---
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'stock_features_dataset.pkl'
    MODEL_CONFIG_FILE = MODELS_DIR / 'stock_model_config.json' # 与个股模型共享配置

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== 开始为个股评分模型准备机器学习数据集 ====="))
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. 高效加载数据
        self.stdout.write("步骤 1/5: 高效加载日线行情和M值历史数据...")
        quotes_df, m_values_series = self._load_data()
        if quotes_df.empty:
            self.stdout.write(self.style.ERROR("错误: 数据库中没有日线行情数据。"))
            return

        # 2. 生成标签
        self.stdout.write(f"步骤 2/5: 生成标签 (模式: {LABEL_CONFIG['mode']}, 向前看: {LABEL_CONFIG['lookforward_days']}天)...")
        labels_df = self._generate_labels(quotes_df)

        # 3. 计算所有股票的因子特征
        self.stdout.write("步骤 3/5: 计算所有股票的因子特征...")
        features_df, feature_names = self._calculate_all_features(quotes_df)

        # 4. 合并特征和标签
        self.stdout.write("步骤 4/5: 合并特征、M值和标签...")
        # 将M值作为一个特征加入
        features_df['market_m_value'] = features_df.index.get_level_values('trade_date').map(m_values_series)
        feature_names.append('market_m_value')

        # 合并所有数据
        final_df = features_df.join(labels_df, how='inner')
        final_df.dropna(inplace=True)

        if final_df.empty:
            self.stdout.write(self.style.ERROR("错误: 合并和清理后，数据集为空。请检查数据范围或因子计算。"))
            return

        # 5. 保存数据集和配置
        self.stdout.write("步骤 5/5: 保存最终数据集和模型配置文件...")
        X = final_df[feature_names]
        y = final_df['label']

        self.stdout.write(f"数据集准备完成。总样本数: {len(X)}")
        self.stdout.write("标签 (label) 统计信息:")
        self.stdout.write(str(y.describe()))

        dataset = {'X': X, 'y': y, 'index': X.index, 'feature_names': feature_names}
        with open(self.DATASET_FILE, 'wb') as f:
            pickle.dump(dataset, f)
        self.stdout.write(self.style.SUCCESS(f"数据集已成功保存至: {self.DATASET_FILE}"))

        model_config = {'feature_names': feature_names}
        with open(self.MODEL_CONFIG_FILE, 'w') as f:
            json.dump(model_config, f, indent=4)
        self.stdout.write(self.style.SUCCESS(f"模型配置文件已成功保存至: {self.MODEL_CONFIG_FILE}"))
        self.stdout.write(self.style.SUCCESS("===== 数据准备流程结束 ====="))

    def _load_data(self):
        """一次性从数据库加载所有需要的数据"""
        # 确定数据加载的起始日期
        first_quote = DailyQuotes.objects.order_by('trade_date').first()
        if not first_quote:
            return pd.DataFrame(), pd.Series()
        
        start_date = first_quote.trade_date + timedelta(days=FACTOR_LOOKBACK_BUFFER)
        self.stdout.write(f"数据加载起始日期 (已考虑因子计算缓冲): {start_date}")

        # 加载日线行情
        quotes_qs = DailyQuotes.objects.filter(trade_date__gte=start_date).values(
            'trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close'
        )
        quotes_df = pd.DataFrame.from_records(quotes_qs)
        
        # 加载M值
        m_values_qs = DailyFactorValues.objects.filter(
            stock_code_id=MARKET_INDICATOR_CODE,
            factor_code_id='dynamic_M_VALUE',
            trade_date__gte=start_date
        ).values('trade_date', 'raw_value')
        m_values_df = pd.DataFrame.from_records(m_values_qs)
        m_values_series = m_values_df.set_index('trade_date')['raw_value'].astype(float)

        return quotes_df, m_values_series

    def _generate_labels(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """根据配置生成标签"""
        df = quotes_df.set_index(['trade_date', 'stock_code_id'])['hfq_close'].unstack()
        
        if LABEL_CONFIG['mode'] == 'return':
            # 计算未来N日收益率
            future_price = df.shift(-LABEL_CONFIG['lookforward_days'])
            labels = (future_price / df) - 1
        elif LABEL_CONFIG['mode'] == 'sharpe':
            # 计算未来N日夏普比率
            returns = df.pct_change()
            daily_rf = (1 + LABEL_CONFIG['risk_free_rate_annual'])**(1/252) - 1
            excess_returns = returns - daily_rf
            
            future_mean = excess_returns.shift(-LABEL_CONFIG['lookforward_days']).rolling(window=LABEL_CONFIG['lookforward_days']).mean()
            future_std = excess_returns.shift(-LABEL_CONFIG['lookforward_days']).rolling(window=LABEL_CONFIG['lookforward_days']).std()
            
            annualized_sharpe = (future_mean / future_std.replace(0, np.nan)) * np.sqrt(252)
            labels = np.tanh(LABEL_CONFIG['tanh_scaling_factor'] * annualized_sharpe)
        else:
            raise ValueError(f"未知的标签模式: {LABEL_CONFIG['mode']}")

        return labels.stack().rename('label').to_frame()

    def _calculate_all_features(self, quotes_df: pd.DataFrame):
        """对所有股票计算所有因子特征"""
        all_features_list = []
        # 从M值模型配置中获取所有需要计算的因子
        try:
            with open(settings.BASE_DIR / 'selection_manager' / 'ml_models' / 'm_value_model_config.json', 'r') as f:
                m_value_config = json.load(f)
            feature_names = m_value_config['feature_names']
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("错误: M值模型配置文件 'm_value_model_config.json' 不存在。请先运行M值模型训练。"))
            # 如果M值模型不存在，则使用一个预定义的因子列表
            feature_names = [
                'dynamic_ADX_CONFIRM', 'dynamic_v2_MA_SLOPE', 'dynamic_v2_MA_SCORE',
                'dynamic_v2_CPC_Factor', 'dynamic_v2_VPCF', 'dynamic_BREAKOUT_PWR',
                'dynamic_VOLUME_SURGE', 'dynamic_MOM_ACCEL', 'dynamic_RSI_OS',
                'dynamic_NEG_DEV', 'dynamic_BOLL_LB', 'dynamic_LOW_VOL',
                'dynamic_MAX_DD', 'dynamic_DOWNSIDE_RISK', 'dynamic_Old_D',
                'dynamic_Old_I', 'dynamic_Old_M'
            ]
            self.stdout.write(self.style.WARNING(f"将使用默认的因子列表: {feature_names}"))

        stock_groups = quotes_df.groupby('stock_code_id')
        
        for stock_code, group_df in tqdm(stock_groups, desc="计算因子特征"):
            group_df = group_df.set_index('trade_date').sort_index()
            
            # 确保数据连续，填充缺失的交易日
            full_date_range = pd.date_range(start=group_df.index.min(), end=group_df.index.max(), freq='B')
            group_df = group_df.reindex(full_date_range).ffill()
            
            if len(group_df) < FACTOR_LOOKBACK_BUFFER:
                continue

            # 使用复用的因子计算器
            calculator = FactorCalculator(group_df)
            stock_features_df = calculator.run(feature_names)
            stock_features_df['stock_code_id'] = stock_code
            all_features_list.append(stock_features_df)

        if not all_features_list:
            return pd.DataFrame(), []
            
        # 合并所有股票的特征
        final_features_df = pd.concat(all_features_list)
        final_features_df = final_features_df.reset_index().rename(columns={'index': 'trade_date'})
        final_features_df = final_features_df.set_index(['trade_date', 'stock_code_id'])
        
        return final_features_df, feature_names

