# selection_manager/service/m_value_service.py

import logging
import numpy as np
import pandas as pd
import json
from django.conf import settings
import tensorflow as tf

logger = logging.getLogger(__name__)

class MValueService:
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    MODEL_FILE = MODELS_DIR / 'csi300_cnn_final_model.h5'
    CONFIG_FILE = MODELS_DIR / 'csi300_model_config.json'
    
    LOOKBACK_WINDOW = 60
    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
    DERIVED_FEATURE_COLS = ['log_return', 'volatility_20d', 'bias_20d', 'bias_60d']
    ALL_FEATURE_COLS = FEATURE_COLS + DERIVED_FEATURE_COLS
    N_FEATURES = len(ALL_FEATURE_COLS)
    
    def __init__(self):
        self._model = None
        self._config = None
        self._dependencies_loaded = False

    def _load_dependencies(self):
        """懒加载模型和配置文件"""
        if not self.MODEL_FILE.exists() or not self.CONFIG_FILE.exists():
            logger.error("模型或配置文件不存在。请先运行Final训练命令。")
            return
        try:
            self._model = tf.keras.models.load_model(self.MODEL_FILE)
            with open(self.CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            logger.info(f"成功加载M值预测模型 [Final] 及配置。")
            logger.info(f"使用的最佳阈值为: {self._config.get('best_threshold')}")
        except Exception as e:
            logger.error(f"加载 [Final] 模型依赖时发生错误: {e}", exc_info=True)
            self._model, self._config = None, None
        self._dependencies_loaded = True

    def _prepare_input_data(self, csi300_df: pd.DataFrame) -> np.ndarray:
        """数据准备逻辑保持不变"""
        if len(csi300_df) < self.LOOKBACK_WINDOW:
            raise ValueError(f"输入数据长度不足，需要{self.LOOKBACK_WINDOW}天，实际{len(csi300_df)}天。")

        df = csi300_df.copy()
        
        for col in self.FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        df['log_return'] = np.log(df['close']).diff().fillna(0)
        df['volatility_20d'] = df['log_return'].rolling(self.LOOKBACK_WINDOW).std().fillna(0)
        ma20 = df['close'].rolling(20).mean()
        ma60 = df['close'].rolling(60).mean()
        df['bias_20d'] = (df['close'] / ma20 - 1).fillna(0)
        df['bias_60d'] = (df['close'] / ma60 - 1).fillna(0)

        feature_window = df.iloc[-self.LOOKBACK_WINDOW:][self.ALL_FEATURE_COLS]

        mean = feature_window.mean()
        std = feature_window.std()
        std[std == 0] = 1e-9
        normalized_window = (feature_window - mean) / std

        return normalized_window.values.reshape(1, self.LOOKBACK_WINDOW, self.N_FEATURES)
    
    def predict_csi300_next_day_trend(self, csi300_df: pd.DataFrame) -> float:
        """
        [Final] 使用单个模型和最佳阈值来计算M值。
        """
        if not self._dependencies_loaded: self._load_dependencies()
        if self._model is None or self._config is None: return 0.0

        try:
            model_input = self._prepare_input_data(csi300_df)
            
            # 预测得到 [P(上涨), P(下跌)]
            pred_proba = self._model.predict(model_input, verbose=0)[0]
            prob_up = pred_proba[0] # 我们只关心上涨的概率

            best_threshold = self._config.get('best_threshold', 0.5)
            logger.info(f"模型预测上涨概率: {prob_up:.2%}, 最佳决策阈值: {best_threshold:.2%}")
            
            # [核心] M值计算公式
            if prob_up >= best_threshold:
                # 概率在阈值之上，映射到 [0, 1]
                # 防止分母为0
                if (1 - best_threshold) == 0: return 1.0
                m_value = (prob_up - best_threshold) / (1 - best_threshold)
            else:
                # 概率在阈值之下，映射到 [-1, 0)
                # 防止分母为0
                if best_threshold == 0: return -1.0
                m_value = (prob_up - best_threshold) / best_threshold
            
            logger.info(f"最终M值: {m_value:.4f}")
            return float(m_value)
            
        except Exception as e:
            logger.error(f"[Final] 预测M值过程中发生错误: {e}", exc_info=True)
            return 0.0

# 创建全局单例
m_value_service_instance = MValueService()
