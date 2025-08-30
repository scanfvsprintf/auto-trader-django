# selection_manager/service/m_value_service.py

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from django.conf import settings

logger = logging.getLogger(__name__)

# ==============================================================================
#  独立的因子计算器 (Decoupled Factor Calculator)
#  [重要] 此类代码与 prepare_m_value_features.py 中的完全一致，以实现解耦。
#  如果因子逻辑更新，需要同步修改这两个地方。
# ==============================================================================
class FactorCalculator:
    """
    一个独立的、解耦的因子计算器。
    它只接收一个标准的OHLCVA DataFrame，并计算所有预定义的因子。
    """
    def __init__(self, df: pd.DataFrame):
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume', 'amount']):
            raise ValueError("输入DataFrame必须包含 'open', 'high', 'low', 'close', 'volume', 'amount' 列")
        self.df = df.copy()
        ta.Imports["verbose"] = False

    def run(self, feature_names: list) -> pd.DataFrame:
        """
        根据给定的特征名称列表，运行所有需要的因子计算。
        """
        all_factors_df = pd.DataFrame(index=self.df.index)
        
        calculator_methods = {
            'dynamic_ADX_CONFIRM': self._calc_adx_confirm,
            'dynamic_v2_MA_SLOPE': self._calc_v2_ma_slope,
            'dynamic_v2_MA_SCORE': self._calc_v2_ma_score,
            'dynamic_v2_CPC_Factor': self._calc_v2_cpc_factor,
            'dynamic_v2_VPCF': self._calc_v2_vpcf,
            'dynamic_BREAKOUT_PWR': self._calc_breakout_pwr,
            'dynamic_VOLUME_SURGE': self._calc_volume_surge,
            'dynamic_MOM_ACCEL': self._calc_mom_accel,
            'dynamic_RSI_OS': self._calc_rsi_os,
            'dynamic_NEG_DEV': self._calc_neg_dev,
            'dynamic_BOLL_LB': self._calc_boll_lb,
            'dynamic_LOW_VOL': self._calc_low_vol,
            'dynamic_MAX_DD': self._calc_max_dd,
            'dynamic_DOWNSIDE_RISK': self._calc_downside_risk,
        }

        for factor_name in feature_names:
            if factor_name in calculator_methods:
                factor_series = calculator_methods[factor_name]()
                all_factors_df[factor_name] = factor_series
            else:
                raise ValueError(f"预测时发现未知特征 '{factor_name}'，模型和数据准备脚本可能不一致。")
        
        return all_factors_df

    # --- 因子计算方法 (与prepare文件完全相同) ---
    def _calc_adx_confirm(self, length=14, adx_threshold=25):
        adx_df = self.df.ta.adx(length=length, high=self.df['high'], low=self.df['low'], close=self.df['close'])
        adx_col, dmp_col, dmn_col = f'ADX_{length}', f'DMP_{length}', f'DMN_{length}'
        condition = (adx_df[adx_col] > adx_threshold) & (adx_df[dmp_col] > adx_df[dmn_col])
        return adx_df[adx_col].where(condition, 0.0).rename('dynamic_ADX_CONFIRM')

    def _calc_v2_ma_slope(self, ma_period=20, ema_period=20):
        ma = self.df['close'].rolling(window=ma_period).mean()
        ma_roc = ma.pct_change(1)
        return ma_roc.ewm(span=ema_period, adjust=False).mean().rename('dynamic_v2_MA_SLOPE')

    def _calc_v2_ma_score(self, p1=5, p2=10, p3=20):
        close = self.df['close']
        ma5 = close.rolling(window=p1).mean()
        ma10 = close.rolling(window=p2).mean()
        ma20 = close.rolling(window=p3).mean()
        spread1 = (close - ma5) / ma5.replace(0, 1e-9)
        spread2 = (ma5 - ma10) / ma10.replace(0, 1e-9)
        spread3 = (ma10 - ma20) / ma20.replace(0, 1e-9)
        return ((spread1 + spread2 + spread3) / 3.0).rename('dynamic_v2_MA_SCORE')

    def _calc_v2_cpc_factor(self, ema_period=10):
        high, low, close = self.df['high'], self.df['low'], self.df['close']
        price_range = high - low
        dcp = (2 * close - high - low) / price_range.replace(0, 1e-9)
        return dcp.ewm(span=ema_period, adjust=False).mean().rename('dynamic_v2_CPC_Factor')

    def _calc_v2_vpcf(self, s=5, l=20, n_smooth=5):
        ma_close_s = self.df['close'].rolling(window=s).mean()
        price_momentum = ma_close_s.pct_change(1)
        ma_amount_s = self.df['amount'].rolling(window=s).mean()
        ma_amount_l = self.df['amount'].rolling(window=l).mean()
        volume_level = (ma_amount_s / ma_amount_l.replace(0, 1e-9)) - 1
        daily_score = price_momentum * volume_level
        return daily_score.ewm(span=n_smooth, adjust=False).mean().rename('dynamic_v2_VPCF')

    def _calc_breakout_pwr(self, lookback=60, atr_period=14):
        high_lookback = self.df['high'].rolling(window=lookback).max().shift(1)
        atr = self.df.ta.atr(length=atr_period, high=self.df['high'], low=self.df['low'], close=self.df['close'])
        return ((self.df['close'] - high_lookback) / atr.replace(0, 1e-9)).rename('dynamic_BREAKOUT_PWR')

    def _calc_volume_surge(self, lookback=20):
        avg_amount = self.df['amount'].rolling(window=lookback).mean().shift(1)
        return (self.df['amount'] / avg_amount.replace(0, 1e-9)).rename('dynamic_VOLUME_SURGE')

    def _calc_mom_accel(self, roc_period=5, shift_period=11):
        roc = self.df['close'].pct_change(roc_period)
        roc_shifted = roc.shift(shift_period)
        return ((roc / roc_shifted.replace(0, np.nan)) - 1).rename('dynamic_MOM_ACCEL')

    def _calc_rsi_os(self, length=14):
        return self.df.ta.rsi(close=self.df['close'], length=length).rename('dynamic_RSI_OS')

    def _calc_neg_dev(self, period=60):
        ma = self.df['close'].rolling(window=period).mean()
        return ((self.df['close'] - ma) / ma.replace(0, 1e-9)).rename('dynamic_NEG_DEV')

    def _calc_boll_lb(self, length=20, std=2.0):
        boll = self.df.ta.bbands(close=self.df['close'], length=length, std=std)
        lower_band = boll[f'BBL_{length}_{std}']
        upper_band = boll[f'BBU_{length}_{std}']
        band_width = upper_band - lower_band
        return ((self.df['close'] - lower_band) / band_width.replace(0, 1e-9)).rename('dynamic_BOLL_LB')

    def _calc_low_vol(self, period=20):
        returns = self.df['close'].pct_change()
        return returns.rolling(window=period).std().rename('dynamic_LOW_VOL')

    def _calc_max_dd(self, period=60):
        rolling_max = self.df['close'].rolling(window=period, min_periods=1).max()
        daily_dd = self.df['close'] / rolling_max - 1.0
        return daily_dd.rolling(window=period, min_periods=1).min().rename('dynamic_MAX_DD')

    def _calc_downside_risk(self, period=60):
        returns = self.df['close'].pct_change()
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        return downside_returns.rolling(window=period).std().rename('dynamic_DOWNSIDE_RISK')

# ==============================================================================
#  重构后的 M 值预测服务 (Refactored M-Value Prediction Service)
# ==============================================================================
class MValueMLService:
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    MODEL_FILE = MODELS_DIR / 'm_value_lgbm_model.joblib'
    CONFIG_FILE = MODELS_DIR / 'm_value_model_config.json'
    
    # 因子计算所需的最长回溯期，应大于所有因子中最大的lookback period
    # 例如，neg_dev(60), max_dd(60)等，给足100天buffer
    REQUIRED_LOOKBACK = 100

    def __init__(self):
        self._model = None
        self._config = None
        self._dependencies_loaded = False

    def _load_dependencies(self):
        """懒加载模型和配置文件"""
        if not self.MODEL_FILE.exists() or not self.CONFIG_FILE.exists():
            logger.error("M值模型或配置文件不存在。请先运行 'prepare_m_value_features' 和 'train_m_value_model' 命令。")
            return
        try:
            self._model = joblib.load(self.MODEL_FILE)
            with open(self.CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            logger.info("成功加载M值预测模型 (LightGBM) 及配置。")
        except Exception as e:
            logger.error(f"加载M值模型依赖时发生错误: {e}", exc_info=True)
            self._model, self._config = None, None
        self._dependencies_loaded = True

    def _prepare_input_data(self, csi300_df: pd.DataFrame) -> pd.DataFrame:
        """
        为单次预测准备特征向量。
        """
        if len(csi300_df) < self.REQUIRED_LOOKBACK:
            raise ValueError(f"输入数据长度不足，需要至少 {self.REQUIRED_LOOKBACK} 天，实际 {len(csi300_df)} 天。")

        df = csi300_df.copy()
        
        # 数据清洗
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.ffill(inplace=True).bfill(inplace=True)

        # 计算所有需要的特征
        feature_names = self._config['feature_names']
        calculator = FactorCalculator(df)
        features_df = calculator.run(feature_names)
        
        # 检查计算后的特征是否有NaN，并返回最后一行的特征向量
        latest_features = features_df.iloc[-1]
        if latest_features.isnull().any():
            # 尝试向前填充，以防万一
            latest_features = features_df.ffill().iloc[-1]
            if latest_features.isnull().any():
                 raise ValueError(f"计算出的最新特征向量包含NaN值，无法进行预测。NaNs in: {latest_features[latest_features.isnull()].index.tolist()}")

        return latest_features[feature_names] # 确保特征顺序与训练时一致

    def predict_csi300_next_day_trend(self, csi300_df: pd.DataFrame) -> float:
        """
        使用重构后的ML模型计算M值。
        M值 = P(牛市) - P(熊市)
        """
        if not self._dependencies_loaded:
            self._load_dependencies()
        
        if self._model is None or self._config is None:
            logger.warning("M值模型未加载，返回中性值 0.0")
            return 0.0

        try:
            # 1. 准备输入特征向量
            feature_vector = self._prepare_input_data(csi300_df)
            
            # 2. 模型预测
            # LightGBM需要 (n_samples, n_features) 的输入
            model_input = feature_vector.values.reshape(1, -1)
            pred_proba = self._model.predict_proba(model_input)[0]
            
            # 假设类别顺序为 0:Bull, 1:Consolidation, 2:Bear
            prob_bull = pred_proba[0]
            prob_consolidation = pred_proba[1]
            prob_bear = pred_proba[2]

            # 3. 计算M值
            m_value = prob_bull - prob_bear
            
            logger.info(
                f"M值预测: P(牛)={prob_bull:.2%}, P(震荡)={prob_consolidation:.2%}, P(熊)={prob_bear:.2%}. "
                f"最终 M-Value = {m_value:.4f}"
            )
            return float(m_value)
        
        except Exception as e:
            logger.error(f"预测M值过程中发生严重错误: {e}", exc_info=True)
            return 0.0

# 创建全局单例，供下游服务导入
m_value_service_instance = MValueMLService()
