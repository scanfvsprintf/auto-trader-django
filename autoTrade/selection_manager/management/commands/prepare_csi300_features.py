# selection_manager/management/commands/prepare_csi300_features.py

import logging
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from django.core.management.base import BaseCommand
from django.conf import settings
from tqdm import tqdm

from common.models import IndexQuotesCsi300

logger = logging.getLogger(__name__)

# ==============================================================================
#  特征配置中心 (Pluggable Feature Configuration)
# ==============================================================================
# 通过修改此字典的True/False值，可以控制哪些特征被用于训练。
# 这是实现“可插拔”设计的核心。
FEATURE_CONFIG = {
    # 趋势动能 (MT) 维度
    'dynamic_ADX_CONFIRM': True,
    'dynamic_v2_MA_SLOPE': True,
    'dynamic_v2_MA_SCORE': True,
    'dynamic_v2_CPC_Factor': True,
    'dynamic_v2_VPCF': True,
    
    # 强势突破 (BO) 维度
    'dynamic_BREAKOUT_PWR': True,
    'dynamic_VOLUME_SURGE': True,
    'dynamic_MOM_ACCEL': True,

    # 均值回归 (MR) 维度
    'dynamic_RSI_OS': True,
    'dynamic_NEG_DEV': True,
    'dynamic_BOLL_LB': True,

    # 质量防御 (QD) 维度
    'dynamic_LOW_VOL': True,
    'dynamic_MAX_DD': True,
    'dynamic_DOWNSIDE_RISK': True,

    # 新因子
    'dynamic_MACD_SIGNAL': False,

    'dynamic_BREAKOUT_DURATION': False,

    # --- 老M值体系因子 ---
    'dynamic_Old_D': True,
    'dynamic_Old_I': True,
    'dynamic_Old_M': True,
}

# ==============================================================================
#  独立的因子计算器 (Decoupled Factor Calculator)
# ==============================================================================
class FactorCalculator:
    """
    一个独立的、解耦的因子计算器。
    它只接收一个标准的OHLCVA DataFrame，并计算所有预定义的因子。
    这个类将被复制到 m_value_service.py 中以保持解耦。
    """
    def __init__(self, df: pd.DataFrame):
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume', 'amount']):
            raise ValueError("输入DataFrame必须包含 'open', 'high', 'low', 'close', 'volume', 'amount' 列")
        self.df = df.copy()
        # 确保 pandas_ta 不会打印不必要的信息
        ta.Imports["verbose"] = False

    def run(self, config: dict) -> (pd.DataFrame, list):
        """
        根据配置运行所有启用的因子计算。
        返回一个包含所有因子值的DataFrame和使用的因子列表。
        """
        all_factors_df = pd.DataFrame(index=self.df.index)
        enabled_factors = []

        # 获取所有可用的计算方法
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
            'dynamic_MACD_SIGNAL':self._calc_macd_signal,
            'dynamic_BREAKOUT_DURATION':self._calc_breakout_duration,
             # --- 新增老M值体系因子 ---
            'dynamic_Old_D': self._calc_old_d,
            'dynamic_Old_I': self._calc_old_i,
            'dynamic_Old_M': self._calc_old_m,
        }

        for factor_name, is_enabled in config.items():
            if is_enabled:
                if factor_name in calculator_methods:
                    logger.debug(f"Calculating factor: {factor_name}")
                    factor_series = calculator_methods[factor_name]()
                    all_factors_df[factor_name] = factor_series
                    enabled_factors.append(factor_name)
                else:
                    logger.warning(f"Factor '{factor_name}' is enabled in config but no calculation method found.")
        
        return all_factors_df, enabled_factors

    # --- 趋势动能 (MT) 因子 ---
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

    # --- 强势突破 (BO) 因子 ---
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

    # --- 均值回归 (MR) 因子 ---
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

    # --- 质量防御 (QD) 因子 ---
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
    
    # --- 新因子 ---
    def _calc_macd_signal(self, fast=12, slow=26, signal=9):
        macd_df = self.df.ta.macd(fast=fast, slow=slow, signal=signal)
        macd_line = macd_df[f'MACD_{fast}_{slow}_{signal}']
        signal_line = macd_df[f'MACDs_{fast}_{slow}_{signal}']
        # 因子定义为：MACD线上穿信号线的强度，且MACD线在0轴之上
        factor = (macd_line - signal_line).where(macd_line > 0, 0)
        return factor.rename('dynamic_MACD_SIGNAL')
    
    def _calc_breakout_duration(self, lookback=20):
        high_lookback = self.df['close'].rolling(window=lookback).max().shift(1)
        is_breakout = self.df['close'] > high_lookback
        
        # 计算连续为True的天数
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




# ==============================================================================
#  主命令 (Main Command)
# ==============================================================================
class Command(BaseCommand):
    help = '[M-Value Refactor] 基于精选因子和连续夏普比率标签，为沪深300指数生成机器学习数据集。'

    # --- 路径和配置 ---
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'm_value_dataset.pkl'
    MODEL_CONFIG_FILE = MODELS_DIR / 'm_value_model_config.json'

    # --- 新标签体系配置 ---
    LABEL_LOOKFORWARD = 3      # 向前看60个交易日计算夏普比率
    RISK_FREE_RATE_ANNUAL = 0.02 # 无风险年利率，设为4%
    TANH_SCALING_FACTOR = 1    # tanh的缩放因子a，使得夏普=1时接近饱和

    def _get_continuous_m_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        根据未来60日年化夏普比率，计算连续的M值标签。
        标签公式: M = tanh(a * SharpeRatio)
        """
        self.stdout.write("步骤 2/4: 应用新的连续M值标签体系 (基于夏普比率)...")
        
        # 计算每日收益率
        returns = df['close'].pct_change()
        
        # 将年化无风险利率转换为日无风险利率
        # (1 + r_annual) = (1 + r_daily)^252  => r_daily = (1 + r_annual)^(1/252) - 1
        # 一年大约有252个交易日
        daily_risk_free_rate = (1 + self.RISK_FREE_RATE_ANNUAL)**(1/252) - 1
        
        # 计算超额收益率
        excess_returns = returns - daily_risk_free_rate
        
        # 使用高效的向量化方法计算未来60天的滚动指标
        # 为了计算未来60天，我们将数据向前移动60天，然后计算过去60天的滚动值
        # 这等价于在当前点计算未来60天的数据
        future_mean = excess_returns.shift(-self.LABEL_LOOKFORWARD).rolling(window=self.LABEL_LOOKFORWARD).mean()
        future_std = excess_returns.shift(-self.LABEL_LOOKFORWARD).rolling(window=self.LABEL_LOOKFORWARD).std()
        
        # 计算年化夏普比率
        # 年化因子 = sqrt(252)
        # 避免除以零
        annualized_sharpe_ratio = (future_mean / future_std.replace(0, np.nan)) * np.sqrt(252)
        
        # 应用tanh函数生成最终的M值标签
        m_labels = np.tanh(self.TANH_SCALING_FACTOR * annualized_sharpe_ratio)
        m_labels_raw = np.tanh(self.TANH_SCALING_FACTOR * annualized_sharpe_ratio)
        m_labels_smoothed = m_labels_raw.rolling(window=5, min_periods=1).mean()
        self.stdout.write(f"标签计算完成。缩放因子a={self.TANH_SCALING_FACTOR}, 无风险利率(年)={self.RISK_FREE_RATE_ANNUAL}")
        
        return m_labels_smoothed

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== [M-Value Refactor] 开始准备机器学习数据集 ====="))
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. 加载并准备数据
        self.stdout.write("步骤 1/4: 加载沪深300行情数据...")
        quotes_qs = IndexQuotesCsi300.objects.all().order_by('trade_date').values()
        df = pd.DataFrame.from_records(quotes_qs)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)

        # 2. 生成标签 (已重构为连续M值)
        labels = self._get_continuous_m_labels(df)
        df['label'] = labels

        # 3. 计算特征
        self.stdout.write("步骤 3/4: 计算所有已启用的特征...")
        calculator = FactorCalculator(df)
        features_df, enabled_features = calculator.run(FEATURE_CONFIG)
        
        # 4. 对齐数据并保存
        self.stdout.write("步骤 4/4: 对齐特征和标签，并保存到文件...")
        
        # 合并特征和标签
        final_df = pd.concat([features_df, df['label']], axis=1)
        
        # 剔除所有包含NaN的行（通常是序列开头和结尾，以及标签计算的未来窗口）
        final_df.dropna(inplace=True)
        
        # 分离 X 和 y
        X = final_df[enabled_features]
        y = final_df['label'].astype(float) # 标签现在是浮点数
        
        self.stdout.write(f"数据集准备完成。总样本数: {len(X)}")
        self.stdout.write("标签（M值）统计信息:")
        self.stdout.write(str(y.describe()))
        
        # 保存数据集和配置
        dataset = {
            'X': X,
            'y': y,
            'index': X.index,
            'feature_names': enabled_features,
            # 'label_map' 不再需要，因为这是回归任务
        }
        with open(self.DATASET_FILE, 'wb') as f:
            pickle.dump(dataset, f)
        
        self.stdout.write(self.style.SUCCESS(f"数据集已成功保存至: {self.DATASET_FILE}"))
        
        # 保存模型配置，主要是特征列表，供预测时使用
        model_config = {'feature_names': enabled_features}
        with open(self.MODEL_CONFIG_FILE, 'w') as f:
            json.dump(model_config, f, indent=4)
        self.stdout.write(self.style.SUCCESS(f"模型配置文件已成功保存至: {self.MODEL_CONFIG_FILE}"))
        self.stdout.write(self.style.SUCCESS("===== [M-Value Refactor] 数据准备流程结束 ====="))
