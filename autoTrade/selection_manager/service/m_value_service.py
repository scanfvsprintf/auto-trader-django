# selection_manager/service/m_value_service.py

import logging
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from django.conf import settings
from decimal import Decimal

logger = logging.getLogger(__name__)

class MValueService:
    """
    使用机器学习模型预测沪深300指数下一日涨跌可能性的服务。
    这是一个采用懒加载单例模式的实现，确保模型只在第一次预测时加载。
    """
    _instance = None
    _model = None
    _model_loaded = False # [修复] 添加一个标志位

    # --- 配置常量 ---
    WINDOW_SIZE = 60
    PREDICTION_THRESHOLD = 0.03
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    MODEL_FILE = MODELS_DIR / 'csi300_lgbm_predictor.joblib'
    
    def __new__(cls):
        # [修复] 简化 __new__，不再加载模型
        if cls._instance is None:
            cls._instance = super(MValueService, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        # [修复] 新增一个专门用于加载模型的方法
        if not self._model_loaded:
            try:
                logger.info(f"正在加载机器学习模型: {self.MODEL_FILE}")
                self._model = joblib.load(self.MODEL_FILE)
                self._model_loaded = True
                logger.info("模型加载成功。")
            except FileNotFoundError:
                logger.error(f"模型文件 {self.MODEL_FILE} 未找到！请先运行 'train_csi300_model' 命令。")
                self._model = None
            except Exception as e:
                logger.error(f"加载模型时发生未知错误: {e}")
                self._model = None
    
    def _extract_features_from_data(self, csi300_60_days_df: pd.DataFrame) -> np.ndarray:
        # ... (此方法与上次修复后的一致，不再重复) ...
        if len(csi300_60_days_df) != self.WINDOW_SIZE:
            raise ValueError(f"输入的数据必须刚好是 {self.WINDOW_SIZE} 天。")
        df = csi300_60_days_df.copy()
        for col in df.columns:
            first_valid_element = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_valid_element, Decimal):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(close=df['close'], length=20, append=True)
        df.ta.atr(high=df['high'], low=df['low'], close=df['close'], append=True)
        df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], append=True)
        df.ta.willr(high=df['high'], low=df['low'], close=df['close'], append=True)
        all_cols = set(df.columns)
        price_cols = {'open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_change', 'change_amount', 'turnover_rate'}
        feature_cols = list(all_cols - price_cols)
        feature_df = df[feature_cols]
        feature_df.fillna(method='ffill', inplace=True)
        feature_df.fillna(method='bfill', inplace=True)
        feature_df.fillna(0, inplace=True)
        return feature_df.values.flatten()


    def predict_csi300_next_day_trend(self, csi300_60_days_df: pd.DataFrame) -> float:
        """
        输入一个包含60天沪深300高开低收数据的DataFrame，预测下一日的涨跌可能性。
        """
        # [修复] 在预测时才检查并加载模型
        if not self._model_loaded:
            self._load_model()

        if self._model is None:
            logger.error("模型未加载或加载失败，无法进行预测。返回中性值0。")
            return 0.0

        try:
            feature_vector = self._extract_features_from_data(csi300_60_days_df)
            raw_prediction = self._model.predict(feature_vector.reshape(1, -1))[0]
            logger.info(f"模型原始预测值 (预期涨跌幅): {raw_prediction:.4%}")
            
            clipped_prediction = np.clip(raw_prediction, -self.PREDICTION_THRESHOLD, self.PREDICTION_THRESHOLD)
            scaled_value = clipped_prediction / self.PREDICTION_THRESHOLD
            
            logger.info(f"缩放后的M值: {scaled_value:.4f}")
            return scaled_value
        except Exception as e:
            logger.error(f"预测过程中发生错误: {e}", exc_info=True)
            return 0.0

# [修复] 仍然创建全局实例，但此时它不会加载模型
m_value_service_instance = MValueService()
