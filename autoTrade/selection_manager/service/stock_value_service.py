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
        获取股票池中所有股票的模型评分。

        :param stock_pool: 待评分的股票代码列表。
        :param trade_date: 评分基准日 (T-1日)。
        :param m_value: 当日的市场M值。
        :return: 一个包含 {stock_code: score} 的 Pandas Series。
        """
        if not self._dependencies_loaded:
            logger.warning("模型未加载，返回空评分列表。")
            return pd.Series(dtype=float)

        # 1. 高效加载所有需要的数据
        start_date = trade_date - timedelta(days=FACTOR_LOOKBACK_BUFFER * 2) # 넉넉하게
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool,
            trade_date__gte=start_date,
            trade_date__lte=trade_date
        ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover')
        
        if not quotes_qs:
            logger.warning("在指定日期范围内未找到任何股票行情数据。")
            return pd.Series(dtype=float)

        all_quotes_df = pd.DataFrame.from_records(quotes_qs)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            all_quotes_df[col] = pd.to_numeric(all_quotes_df[col], errors='coerce')
        # 2. 并行计算特征和预测
        scores = {}
        # 使用tqdm包装ThreadPoolExecutor以显示进度条
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_stock = {
                executor.submit(self._predict_single_stock, stock_code, group_df, m_value): stock_code
                for stock_code, group_df in all_quotes_df.groupby('stock_code_id')
            }
            
            progress = tqdm(as_completed(future_to_stock), total=len(stock_pool), desc="计算个股评分")
            for future in progress:
                stock_code = future_to_stock[future]
                try:
                    score = future.result()
                    if score is not None:
                        scores[stock_code] = score
                except Exception as exc:
                    logger.error(f"为股票 {stock_code} 计算评分时出错: {exc}")

        return pd.Series(scores)

    def _predict_single_stock(self, stock_code: str, stock_df: pd.DataFrame, m_value: float) -> float | None:
        """
        为单只股票计算特征并进行预测。
        """
        # 准备数据
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

