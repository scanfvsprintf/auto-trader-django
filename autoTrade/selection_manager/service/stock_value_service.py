# ==============================================================================
# 文件 3/4: selection_manager/service/stock_value_service.py
# 描述: 提供个股模型评分的服务，衔接 selection_service.py。
# ==============================================================================
import logging
import json
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from tqdm import tqdm

from common.models import DailyQuotes
from .m_value_service import FactorCalculator # 复用因子计算器

logger = logging.getLogger(__name__)

# 因子计算所需的最大回溯期
FACTOR_LOOKBACK_BUFFER = 100

class StockValueService:
    """
    个股价值评分服务。
    加载训练好的机器学习模型，为给定的股票池计算当日的模型评分。
    """
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    MODEL_FILE = MODELS_DIR / 'stock_lgbm_model.joblib'
    CONFIG_FILE = MODELS_DIR / 'stock_model_config.json'

    def __init__(self):
        self._model = None
        self._config = None
        self._feature_names = []
        self._dependencies_loaded = False
        self._load_dependencies()

    def _load_dependencies(self):
        """懒加载模型和配置文件"""
        if not self.MODEL_FILE.exists() or not self.CONFIG_FILE.exists():
            logger.error("个股评分模型或配置文件不存在。请先运行 'prepare_stock_features' 和 'train_stock_model' 命令。")
            return
        try:
            self._model = joblib.load(self.MODEL_FILE)
            with open(self.CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            self._feature_names = self._config['feature_names']
            logger.info("成功加载个股评分模型 (LightGBM Regressor) 及配置。")
            self._dependencies_loaded = True
        except Exception as e:
            logger.error(f"加载个股评分模型依赖时发生错误: {e}", exc_info=True)
            self._model, self._config = None, None

    def get_all_stock_scores(self, stock_pool: list, trade_date, m_value: float) -> pd.Series:
        """
        [高效向量化版] 获取股票池中所有股票的模型评分。
        """
        if not self._dependencies_loaded:
            logger.warning("模型未加载，返回空评分列表。")
            return pd.Series(dtype=float)
        # 1. 高效加载所有需要的数据
        start_date = trade_date - timedelta(days=FACTOR_LOOKBACK_BUFFER * 2)
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool,
            trade_date__gte=start_date,
            trade_date__lte=trade_date
        ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover')
        
        if not quotes_qs:
            logger.warning("在指定日期范围内未找到任何股票行情数据。")
            return pd.Series(dtype=float)
        all_quotes_df = pd.DataFrame.from_records(quotes_qs)
        
        # 预处理：类型转换和列名统一
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            all_quotes_df[col] = pd.to_numeric(all_quotes_df[col], errors='coerce')
        all_quotes_df.rename(columns={'turnover': 'amount'}, inplace=True)
        # 2. 构建面板数据 (Panel Data)
        # 这是向量化计算的基础
        logger.debug("构建面板数据...")
        panel_data = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            panel = all_quotes_df.pivot(index='trade_date', columns='stock_code_id', values=col)
            # 确保数据连续，并向前填充
            panel = panel.reindex(pd.date_range(start=panel.index.min(), end=panel.index.max(), freq='B')).ffill()
            panel_data[col] = panel
        # 3. 一次性计算所有股票的因子特征
        # 我们需要一个修改版的因子计算器，让它能处理面板数据
        # 但为了最小改动，我们这里模拟一个可以处理面板的简单循环
        # 注意：理想的FactorCalculator也应该是全向量化的，但这里我们先解决主要矛盾
        logger.debug("开始向量化计算所有因子...")
        
        # 创建一个临时的、包含所有面板数据的DataFrame，以适配现有的FactorCalculator
        # 这不是最高效的，但避免了修改FactorCalculator
        all_features_list = []
        
        # 这里的循环是在列上（股票代码），而不是在行上，数量少得多
        for stock_code in tqdm(panel_data['close'].columns, desc="计算因子特征"):
            # 为每只股票构建一个符合FactorCalculator输入的DataFrame
            stock_df = pd.DataFrame({
                'open': panel_data['open'][stock_code],
                'high': panel_data['high'][stock_code],
                'low': panel_data['low'][stock_code],
                'close': panel_data['close'][stock_code],
                'volume': panel_data['volume'][stock_code],
                'amount': panel_data['amount'][stock_code],
            }).dropna(how='all') # 删除完全是NaN的行
            if len(stock_df) < FACTOR_LOOKBACK_BUFFER:
                continue
            calculator = FactorCalculator(stock_df)
            features_to_calc = [f for f in self._feature_names if f != 'market_m_value']
            stock_features_df = calculator.run(features_to_calc)
            
            # 只取最后一天的特征
            latest_features = stock_features_df.iloc[-1].copy()
            latest_features.name = stock_code # 将Series的name设置为股票代码
            all_features_list.append(latest_features)
        if not all_features_list:
            logger.warning("未能为任何股票计算出有效的特征。")
            return pd.Series(dtype=float)
        # 将所有股票的最新特征合并成一个DataFrame
        features_df = pd.concat(all_features_list, axis=1).T
        features_df.index.name = 'stock_code_id'
        # 4. 准备模型输入
        logger.debug("准备模型输入并进行预测...")
        # 添加M值特征
        features_df['market_m_value'] = m_value
        
        # 剔除任何包含NaN的行
        features_df.dropna(inplace=True)
        
        if features_df.empty:
            logger.warning("所有股票在特征计算后都因NaN被剔除。")
            return pd.Series(dtype=float)
        # 保证特征顺序
        model_input = features_df[self._feature_names]
        
        # 5. 一次性预测所有股票
        scores = self._model.predict(model_input)
        
        # 6. 组装成最终的Series
        final_scores = pd.Series(scores, index=model_input.index)
        
        logger.info(f"成功为 {len(final_scores)} 只股票计算了模型评分。")
        return final_scores

    def _predict_single_stock(self, stock_code: str, stock_df: pd.DataFrame, m_value: float) -> float | None:
        """
        为单只股票计算特征并进行预测。
        """
        # 准备数据
        stock_df = stock_df.rename(columns={'turnover': 'amount'})
        stock_df = stock_df.set_index('trade_date').sort_index()
        
        # 确保数据连续
        full_date_range = pd.date_range(start=stock_df.index.min(), end=stock_df.index.max(), freq='B')
        stock_df = stock_df.reindex(full_date_range).ffill()

        if len(stock_df) < FACTOR_LOOKBACK_BUFFER:
            return None

        # 计算因子
        calculator = FactorCalculator(stock_df)
        # 从配置中移除 'market_m_value'，因为它不是由计算器生成的
        features_to_calc = [f for f in self._feature_names if f != 'market_m_value']
        features_df = calculator.run(features_to_calc)
        
        # 获取最后一行的特征
        latest_features = features_df.iloc[-1].copy()
        
        # 添加M值特征
        latest_features['market_m_value'] = m_value
        
        # 检查NaN
        if latest_features.isnull().any():
            # logger.warning(f"股票 {stock_code} 计算出的最新特征向量包含NaN值，无法预测。")
            return None
            
        # 保证特征顺序
        model_input = latest_features[self._feature_names].to_frame().T
        
        # 预测
        score = self._model.predict(model_input)[0]
        return float(score)

