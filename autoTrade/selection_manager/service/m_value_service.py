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
#  [重要] 此类代码与 prepare_csi300_features.py 中的完全一致，以实现解耦。
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
        
        # [同步] 确保此处的计算方法列表与 prepare_csi300_features.py 完全一致
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
            'dynamic_MACD_SIGNAL': self._calc_macd_signal,
            'dynamic_BREAKOUT_DURATION': self._calc_breakout_duration,
            # --- 新增老M值体系因子 ---
            'dynamic_Old_D': self._calc_old_d,
            'dynamic_Old_I': self._calc_old_i,
            'dynamic_Old_M': self._calc_old_m,
            'avg_amount_5d': self._calc_avg_amount_5d
        }

        for factor_name in feature_names:
            if factor_name in calculator_methods:
                factor_series = calculator_methods[factor_name]()
                all_factors_df[factor_name] = factor_series
            else:
                #raise ValueError(f"预测时发现未知特征 '{factor_name}'，模型和数据准备脚本可能不一致。")
                pass
        
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

    # [同步] 补全与 prepare_csi300_features.py 一致的因子
    def _calc_macd_signal(self, fast=12, slow=26, signal=9):
        macd_df = self.df.ta.macd(fast=fast, slow=slow, signal=signal)
        macd_line = macd_df[f'MACD_{fast}_{slow}_{signal}']
        signal_line = macd_df[f'MACDs_{fast}_{slow}_{signal}']
        factor = (macd_line - signal_line).where(macd_line > 0, 0)
        return factor.rename('dynamic_MACD_SIGNAL')
    
    def _calc_breakout_duration(self, lookback=20):
        high_lookback = self.df['close'].rolling(window=lookback).max().shift(1)
        is_breakout = self.df['close'] > high_lookback
        breakout_streaks = is_breakout.groupby((is_breakout != is_breakout.shift()).cumsum()).cumsum()
        return breakout_streaks.rename('dynamic_BREAKOUT_DURATION')
    

    def _calc_old_d(self, lookback_k=20, a_param=200.0):
        """
        计算老M值体系中的方向函数 D(t)。
        D(t) = tanh(a * h(t,K))
        h(t,K) 是过去K天不同周期线性回归斜率的均值。
        """
        from scipy.stats import linregress # 仅在此方法中需要
        close_prices = self.df['close']
        g_values_list = []
        
        # 为了向量化计算，我们创建一个包含所有需要回归的窗口的DataFrame
        # 对于每个交易日t，我们需要计算从t-k到t的回归，k从1到lookback_k
        for k in range(1, lookback_k + 1):
            # 截取 k+1 个数据点
            windows = close_prices.rolling(window=k + 1)
            
            # 使用apply函数对每个窗口进行线性回归
            # apply函数会比较慢，但对于这种复杂的窗口计算是必要的
            # 注意：linregress需要numpy数组
            slopes = windows.apply(lambda x: linregress(np.arange(len(x)), x).slope, raw=True)
            
            # 获取 t-k 日的收盘价
            close_t_minus_k = close_prices.shift(k)
            
            # 计算 g(t,k)
            g_tk = slopes / close_t_minus_k.replace(0, 1e-9)
            g_values_list.append(g_tk)
        # 将所有g(t,k)的值合并成一个DataFrame
        g_df = pd.concat(g_values_list, axis=1)
        
        # 计算 h(t,K)，即对每一行（每个交易日）的g值求均值
        h_t_k = g_df.mean(axis=1)
        
        # 计算 D(t)
        d_t = np.tanh(a_param * h_t_k)
        
        return d_t.rename('dynamic_Old_D')
    def _calc_old_i(self, adx_period=14, adx_threshold=20.0, b_param=0.075):
        """
        计算老M值体系中的强度函数 I(t)。
        I(t) = max(0, tanh(b * (ADX(t) - threshold)))
        """
        # pandas_ta库可以非常高效地计算ADX
        adx_df = self.df.ta.adx(length=adx_period, high=self.df['high'], low=self.df['low'], close=self.df['close'])
        adx_series = adx_df[f'ADX_{adx_period}']
        
        # 计算 I(t)
        raw_i = np.tanh(b_param * (adx_series - adx_threshold))
        i_t = raw_i.clip(lower=0) # 使用clip实现max(0, raw_i)
        
        return i_t.rename('dynamic_Old_I')
    def _calc_old_m(self):
        """
        计算老M值 OldM(t) = D(t) * I(t)。
        这个因子依赖于 _calc_old_d 和 _calc_old_i 的计算结果。
        为了效率，我们直接在这里调用它们，而不是重复计算。
        """
        # 为了避免重复计算，我们检查这些列是否已存在于一个临时的DataFrame中
        # 但在当前架构下，最简单的做法是重新计算一次，或者修改run方法
        # 这里我们选择直接计算，因为因子计算是独立的
        d_t = self._calc_old_d()
        i_t = self._calc_old_i()
        
        m_t = d_t * i_t
        
        return m_t.rename('dynamic_Old_M')
    def _calc_avg_amount_5d(self, period=5):
        """计算N日平均成交额"""
        return self.df['amount'].rolling(window=period).mean().rename('avg_amount_5d')

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
            logger.error("M值模型或配置文件不存在。请先运行 'prepare_csi300_features' 和 'train_csi300_model_test' 命令。")
            return
        try:
            self._model = joblib.load(self.MODEL_FILE)
            with open(self.CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            logger.info("成功加载M值预测模型 (LightGBM Regressor) 及配置。")
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
        #df.ffill(inplace=True).bfill(inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

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
        使用重构后的ML回归模型直接预测M值。
        M值由模型直接输出，其范围在[-1, 1]之间。
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
            # 回归模型直接输出预测值
            model_input = feature_vector.to_frame().T 
            m_value = self._model.predict(model_input)[0]
            
            # 3. 裁剪M值确保在[-1, 1]范围内，以防模型预测略微超限
            m_value = np.clip(m_value, -1.0, 1.0)
            
            logger.info(
                f"M值预测: 模型直接输出 M-Value = {m_value:.4f}"
            )
            return float(m_value)
        
        except Exception as e:
            logger.error(f"预测M值过程中发生严重错误: {e}", exc_info=True)
            return 0.0

# 创建全局单例，供下游服务导入
m_value_service_instance = MValueMLService()
