# selection_manager/service/stock_factor_calculator.py
# [新增文件]
# 描述: 个股模型专用的因子计算引擎。
#       该文件复制并扩展了 m_value_service.py 中的 FactorCalculator，
#       以实现与M值模型计算逻辑的完全隔离。
#       所有个股模型的新增或修改因子，都应在此文件中进行。

import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class StockFactorCalculator:
    """
    一个独立的、解耦的、向量化的个股因子计算器。
    它接收面板数据，并计算所有为个股模型定义的因子。
    """
    def __init__(self, panel_data: dict, feature_names: list):
        """
        初始化因子计算器。
        :param panel_data: 一个包含 'open', 'high', 'low', 'close', 'volume', 'amount' 面板数据的字典。
                           每个值都是一个 (日期 x 股票) 的 DataFrame。
        :param feature_names: 需要计算的因子名称列表。
        """
        required_panels = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(p in panel_data for p in required_panels):
            raise ValueError(f"输入 panel_data 必须包含以下键: {required_panels}")
            
        self.open = panel_data['open']
        self.high = panel_data['high']
        self.low = panel_data['low']
        self.close = panel_data['close']
        self.volume = panel_data['volume']
        self.amount = panel_data['amount']
        
        # 新增因子可能需要的数据
        self.m_value_series = panel_data.get('m_value_series')
        self.corp_action_panel = panel_data.get('corp_action_panel')

        self.feature_names = feature_names
        self.epsilon = 1e-9 # 用于防止除以零的小常数

    def run(self) -> pd.DataFrame:
        """
        根据配置运行所有启用的因子计算。
        返回一个包含所有因子值的DataFrame，索引为 (trade_date, stock_code_id)。
        """
        # 因子计算方法的映射字典
        calculator_methods = {
            # --- 原有M值模型复用的因子 ---
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
            'dynamic_Old_D': self._calc_old_d,
            'dynamic_Old_I': self._calc_old_i,
            'dynamic_Old_M': self._calc_old_m,
            # --- 本次新增的因子 ---
            'market_m_value_lag1': self._calc_market_m_value_lag1,
            'm_value_slope_5d': self._calc_m_value_slope_5d,
            'm_value_slope_20d': self._calc_m_value_slope_20d,
            'corporate_action_factor': self._calc_corporate_action_factor,
            'turnover_slope_3d': self._calc_turnover_slope_3d,
            'turnover_slope_10d': self._calc_turnover_slope_10d,
            'avg_turnover_5d': self._calc_avg_turnover_5d,
        }

        all_factors_panels = {}
        for factor_name in self.feature_names:
            # market_m_value 是外部直接传入的，不是计算出来的
            if factor_name == 'market_m_value':
                continue
            if factor_name in calculator_methods:
                logger.debug(f"Calculating stock factor: {factor_name}")
                # 每个计算方法返回一个 (日期 x 股票) 的面板
                factor_panel = calculator_methods[factor_name]()
                all_factors_panels[factor_name] = factor_panel
            else:
                logger.warning(f"Factor '{factor_name}' is requested but no calculation method found in StockFactorCalculator.")
        
        # 将所有因子面板合并为一个大的多列DataFrame
        if not all_factors_panels:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_factors_panels, axis=1)
        # 将宽格式 (日期 x (因子, 股票)) 转换为长格式 ( (日期, 股票) x 因子 )
        # stack() 默认会将最内层的列索引转换为行索引
        long_format_df = combined_df.stack(level=1)
        long_format_df.index.names = ['trade_date', 'stock_code']
        
        return long_format_df

    # === 辅助函数 ===
    def _calculate_tr(self):
        """[内部辅助函数] 统一计算真实波幅 (True Range)"""
        tr1 = self.high - self.low
        tr2 = abs(self.high - self.close.shift(1))
        tr3 = abs(self.low - self.close.shift(1))
        # 使用 np.maximum 逐元素比较三个DataFrame
        return pd.DataFrame(np.maximum(tr1, np.maximum(tr2, tr3)), index=self.close.index, columns=self.close.columns)

    def _rolling_regression_slope(self, panel_data: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        对面板数据（每一列是一个时间序列）进行高效的滚动线性回归斜率计算。
        :param panel_data: (日期 x 股票) 的 DataFrame。
        :param window: 滚动窗口大小。
        :return: 一个与输入形状相同的 DataFrame，值为斜率。
        """
        # 时间变量 x 是固定的
        x = np.arange(window)
        x_mean = x.mean()
        x_ss = np.sum((x - x_mean)**2)

        # 使用 apply 函数，虽然不是最快的，但对于面板数据是最直接和健壮的
        # 这里的 apply 是对每个滚动窗口（一个Series）进行操作
        def get_slope(y):
            # 确保 y 是 numpy 数组并且没有 NaN
            y_val = y.values
            if np.isnan(y_val).any():
                return np.nan
            # 使用 numpy 的 polyfit 进行线性拟合，只取斜率
            return np.polyfit(x, y_val, 1)[0]

        # 对整个面板进行滚动和应用
        slopes_panel = panel_data.rolling(window=window).apply(get_slope, raw=False)
        return slopes_panel

    # === 原有因子计算 (向量化版本) ===
    def _calc_adx_confirm(self, length=14, adx_threshold=25):
        move_up = self.high.diff()
        move_down = -self.low.diff()
        plus_dm = pd.DataFrame(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=self.high.index, columns=self.high.columns)
        minus_dm = pd.DataFrame(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=self.low.index, columns=self.low.columns)
        
        tr = self._calculate_tr()
        alpha = 1 / length
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        
        di_sum = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / (di_sum + self.epsilon))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        condition = (adx > adx_threshold) & (plus_di > minus_di)
        return adx.where(condition, 0.0)

    def _calc_v2_ma_slope(self, ma_period=20, ema_period=20):
        ma = self.close.rolling(window=ma_period).mean()
        ma_roc = ma.pct_change(1,fill_method=None)
        return ma_roc.ewm(span=ema_period, adjust=False).mean()

    def _calc_v2_ma_score(self, p1=5, p2=10, p3=20):
        ma5 = self.close.rolling(window=p1).mean()
        ma10 = self.close.rolling(window=p2).mean()
        ma20 = self.close.rolling(window=p3).mean()
        spread1 = (self.close - ma5) / (ma5 + self.epsilon)
        spread2 = (ma5 - ma10) / (ma10 + self.epsilon)
        spread3 = (ma10 - ma20) / (ma20 + self.epsilon)
        return (spread1 + spread2 + spread3) / 3.0

    def _calc_v2_cpc_factor(self, ema_period=10):
        price_range = self.high - self.low
        dcp = (2 * self.close - self.high - self.low) / (price_range + self.epsilon)
        return dcp.ewm(span=ema_period, adjust=False).mean()

    def _calc_v2_vpcf(self, s=5, l=20, n_smooth=5):
        ma_close_s = self.close.rolling(window=s).mean()
        price_momentum = ma_close_s.pct_change(1,fill_method=None)
        ma_amount_s = self.amount.rolling(window=s).mean()
        ma_amount_l = self.amount.rolling(window=l).mean()
        volume_level = (ma_amount_s / (ma_amount_l + self.epsilon)) - 1
        daily_score = price_momentum * volume_level
        return daily_score.ewm(span=n_smooth, adjust=False).mean()

    def _calc_breakout_pwr(self, lookback=60, atr_period=14):
        high_lookback = self.high.rolling(window=lookback).max().shift(1)
        tr = self._calculate_tr()
        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
        return (self.close - high_lookback) / (atr + self.epsilon)

    def _calc_volume_surge(self, lookback=20):
        avg_amount = self.amount.rolling(window=lookback).mean().shift(1)
        return self.amount / (avg_amount + self.epsilon)

    def _calc_mom_accel(self, roc_period=5, shift_period=11):
        roc = self.close.pct_change(roc_period,fill_method=None)
        roc_shifted = roc.shift(shift_period)
        return (roc / (roc_shifted + self.epsilon)) - 1

    def _calc_rsi_os(self, length=14):
        delta = self.close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + self.epsilon)
        return 100 - (100 / (1 + rs))

    def _calc_neg_dev(self, period=60):
        ma = self.close.rolling(window=period).mean()
        return (self.close - ma) / (ma + self.epsilon)

    def _calc_boll_lb(self, length=20, std=2.0):
        ma = self.close.rolling(window=length).mean()
        rolling_std = self.close.rolling(window=length).std()
        upper_band = ma + (rolling_std * std)
        lower_band = ma - (rolling_std * std)
        band_width = upper_band - lower_band
        return (self.close - lower_band) / (band_width + self.epsilon)

    def _calc_low_vol(self, period=20):
        returns = self.close.pct_change(fill_method=None)
        return returns.rolling(window=period).std()

    def _calc_max_dd(self, period=60):
        rolling_max = self.close.rolling(window=period, min_periods=1).max()
        daily_dd = self.close / rolling_max - 1.0
        return daily_dd.rolling(window=period, min_periods=1).min()

    def _calc_downside_risk(self, period=60):
        returns = self.close.pct_change(fill_method=None)
        downside_returns = returns.clip(upper=0)
        return downside_returns.rolling(window=period).std()

    def _calc_old_d(self, lookback_k=20, a_param=200.0):
        # This factor is slow and not easily vectorized. We calculate it per stock.
        # It's a good candidate for future optimization (e.g., with Numba).
        all_slopes = {}
        for stock_code in self.close.columns:
            series = self.close[stock_code].dropna()
            if len(series) < lookback_k + 1:
                all_slopes[stock_code] = pd.Series(np.nan, index=series.index)
                continue
            
            h_t_k_list = []
            for k in range(1, lookback_k + 1):
                windows = series.rolling(window=k + 1)
                slopes = windows.apply(lambda x: linregress(np.arange(len(x)), x).slope if len(x) == k+1 else np.nan, raw=True)
                close_t_minus_k = series.shift(k)
                g_tk = slopes / (close_t_minus_k + self.epsilon)
                h_t_k_list.append(g_tk)
            
            h_df = pd.concat(h_t_k_list, axis=1)
            h_t_k = h_df.mean(axis=1)
            all_slopes[stock_code] = np.tanh(a_param * h_t_k)
            
        return pd.DataFrame(all_slopes)

    def _calc_old_i(self, adx_period=14, adx_threshold=20.0, b_param=0.075):
        # Re-using the ADX calculation from _calc_adx_confirm
        move_up = self.high.diff()
        move_down = -self.low.diff()
        plus_dm = pd.DataFrame(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=self.high.index, columns=self.high.columns)
        minus_dm = pd.DataFrame(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=self.low.index, columns=self.low.columns)
        
        tr = self._calculate_tr()
        alpha = 1 / adx_period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        di_sum = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / (di_sum + self.epsilon))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        raw_i = np.tanh(b_param * (adx - adx_threshold))
        return raw_i.clip(lower=0)

    def _calc_old_m(self):
        d_t = self._calc_old_d()
        i_t = self._calc_old_i()
        return d_t * i_t

    # === 新增因子计算 ===
    def _calc_market_m_value_lag1(self):
        if self.m_value_series is None or self.m_value_series.empty:
            return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)
        
        m_lagged = self.m_value_series.shift(1)
        
        # --- [推荐修改] ---
        # 使用Pandas的广播机制，更简洁、更安全
        # 创建一个与目标面板形状相同的空DataFrame，然后按行（axis=0）加上该序列
        return pd.DataFrame(dtype=float,index=self.close.index, columns=self.close.columns).add(m_lagged, axis=0)
    def _calc_m_value_slope_5d(self):
        if self.m_value_series is None or self.m_value_series.empty:
            return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)
        
        slopes = self._rolling_regression_slope(self.m_value_series.to_frame(), window=5).iloc[:, 0]
        # --- [推荐修改] ---
        return pd.DataFrame(dtype=float,index=self.close.index, columns=self.close.columns).add(slopes, axis=0)
    def _calc_m_value_slope_20d(self):
        if self.m_value_series is None or self.m_value_series.empty:
            return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)
            
        slopes = self._rolling_regression_slope(self.m_value_series.to_frame(), window=20).iloc[:, 0]
        # --- [推荐修改] ---
        return pd.DataFrame(dtype=float,index=self.close.index, columns=self.close.columns).add(slopes, axis=0)

    def _calc_corporate_action_factor(self):
        if self.corp_action_panel is None or self.corp_action_panel.empty:
            return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)
        
        # The panel is already calculated in the prepare step.
        # Here we just need to forward fill for 10 days.
        factor_panel = self.corp_action_panel.replace(0, np.nan).ffill(limit=9).fillna(0)
        return factor_panel

    def _calc_turnover_slope_3d(self):
        return self._rolling_regression_slope(self.amount, window=3)

    def _calc_turnover_slope_10d(self):
        return self._rolling_regression_slope(self.amount, window=10)

    def _calc_avg_turnover_5d(self):
        return self.amount.rolling(window=5).mean()
