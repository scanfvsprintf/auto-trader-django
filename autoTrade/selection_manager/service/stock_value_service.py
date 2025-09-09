# ==============================================================================
# 文件: selection_manager/service/stock_value_service.py (终极性能优化版)
# 描述: 提供个股模型评分的服务，使用完全向量化的因子计算引擎。
# ==============================================================================
import logging
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from scipy.stats import linregress

from common.models import DailyQuotes

logger = logging.getLogger(__name__)

# 因子计算所需的最大回溯期
FACTOR_LOOKBACK_BUFFER = 100

# ==============================================================================
#  高效向量化因子计算引擎 (High-Performance Vectorized Factor Engine)
# ==============================================================================
class VectorizedFactorEngine:
    """
    一个完全向量化的因子计算引擎。
    它接收面板数据，并一次性计算所有股票的因子值。
    """
    def __init__(self, panel_data: dict, feature_names: list):
        self.open = panel_data['open']
        self.high = panel_data['high']
        self.low = panel_data['low']
        self.close = panel_data['close']
        self.volume = panel_data['volume']
        self.amount = panel_data['amount']
        self.feature_names = feature_names
        self.epsilon = 1e-9
    def run(self) -> pd.DataFrame:
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
            'dynamic_Old_D': self._calc_old_d,
            'dynamic_Old_I': self._calc_old_i,
            'dynamic_Old_M': self._calc_old_m,
            'avg_amount_5d': self._calc_avg_amount_5d,
            'dynamic_TD_COUNT': self._calc_td_count
        }
        all_factors = {}
        for factor_name in self.feature_names:
            if factor_name in calculator_methods:
                logger.debug(f"Vectorized calculation for: {factor_name}")
                all_factors[factor_name] = calculator_methods[factor_name]()
        return pd.DataFrame(all_factors)
    def _calculate_tr(self):
        """[内部辅助函数] 统一计算真实波幅 (True Range)"""
        tr1 = self.high - self.low
        tr2 = abs(self.high - self.close.shift(1))
        tr3 = abs(self.low - self.close.shift(1))
        return np.maximum(tr1, np.maximum(tr2, tr3))
    def _calc_adx_confirm(self, length=14, adx_threshold=25):
        move_up = self.high.diff()
        move_down = -self.low.diff()
        plus_dm = pd.DataFrame(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=self.high.index, columns=self.high.columns)
        minus_dm = pd.DataFrame(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=self.low.index, columns=self.low.columns)
        
        tr = self._calculate_tr() # 调用统一的TR计算函数
        alpha = 1 / length
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        
        di_sum = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / di_sum.replace(0, np.inf))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        condition = (adx.iloc[-1] > adx_threshold) & (plus_di.iloc[-1] > minus_di.iloc[-1])
        return adx.iloc[-1].where(condition, 0.0)
    def _calc_v2_ma_slope(self, ma_period=20, ema_period=20):
        ma = self.close.rolling(window=ma_period).mean()
        ma_roc = ma.pct_change(1)
        return ma_roc.ewm(span=ema_period, adjust=False).mean().iloc[-1]
    def _calc_v2_ma_score(self, p1=5, p2=10, p3=20):
        ma5 = self.close.rolling(window=p1).mean()
        ma10 = self.close.rolling(window=p2).mean()
        ma20 = self.close.rolling(window=p3).mean()
        spread1 = (self.close - ma5) / (ma5 + self.epsilon)
        spread2 = (ma5 - ma10) / (ma10 + self.epsilon)
        spread3 = (ma10 - ma20) / (ma20 + self.epsilon)
        return ((spread1 + spread2 + spread3) / 3.0).iloc[-1]
    def _calc_v2_cpc_factor(self, ema_period=10):
        price_range = self.high - self.low
        dcp = (2 * self.close - self.high - self.low) / (price_range + self.epsilon)
        return dcp.ewm(span=ema_period, adjust=False).mean().iloc[-1]
    def _calc_v2_vpcf(self, s=5, l=20, n_smooth=5):
        ma_close_s = self.close.rolling(window=s).mean()
        price_momentum = ma_close_s.pct_change(1)
        ma_amount_s = self.amount.rolling(window=s).mean()
        ma_amount_l = self.amount.rolling(window=l).mean()
        volume_level = (ma_amount_s / (ma_amount_l + self.epsilon)) - 1
        daily_score = price_momentum * volume_level
        return daily_score.ewm(span=n_smooth, adjust=False).mean().iloc[-1]
    def _calc_breakout_pwr(self, lookback=60, atr_period=14):
        high_lookback = self.high.rolling(window=lookback).max().shift(1)
        
        tr = self._calculate_tr() # 调用统一的TR计算函数
        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
        
        return ((self.close - high_lookback) / (atr + self.epsilon)).iloc[-1]
    def _calc_volume_surge(self, lookback=20):
        avg_amount = self.amount.rolling(window=lookback).mean().shift(1)
        return (self.amount / (avg_amount + self.epsilon)).iloc[-1]
    def _calc_mom_accel(self, roc_period=5, shift_period=11):
        roc = self.close.pct_change(roc_period)
        roc_shifted = roc.shift(shift_period)
        # 使用 where 避免分母为0时产生 inf
        acceleration = (roc / roc_shifted).where(roc_shifted != 0, 1) - 1
        return acceleration.iloc[-1]
    def _calc_rsi_os(self, length=14):
        delta = self.close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + self.epsilon)
        return (100 - (100 / (1 + rs))).iloc[-1]
    def _calc_neg_dev(self, period=60):
        ma = self.close.rolling(window=period).mean()
        return ((self.close - ma) / (ma + self.epsilon)).iloc[-1]
    def _calc_boll_lb(self, length=20, std=2.0):
        ma = self.close.rolling(window=length).mean()
        rolling_std = self.close.rolling(window=length).std()
        upper_band = ma + (rolling_std * std)
        lower_band = ma - (rolling_std * std)
        band_width = upper_band - lower_band
        return ((self.close - lower_band) / (band_width + self.epsilon)).iloc[-1]
    def _calc_low_vol(self, period=20):
        returns = self.close.pct_change()
        return returns.rolling(window=period).std().iloc[-1]
    def _calc_max_dd(self, period=60):
        rolling_max = self.close.rolling(window=period, min_periods=1).max()
        daily_dd = self.close / rolling_max - 1.0
        return daily_dd.rolling(window=period, min_periods=1).min().iloc[-1]
    def _calc_downside_risk(self, period=60):
        returns = self.close.pct_change()
        downside_returns = returns.clip(upper=0)
        return downside_returns.rolling(window=period).std().iloc[-1]
    def _calc_old_d(self, lookback_k=20, a_param=200.0):
        slopes = {}
        x_range = np.arange(lookback_k + 1)
        for stock_code in self.close.columns:
            series = self.close[stock_code].dropna()
            if len(series) < lookback_k + 1:
                slopes[stock_code] = np.nan
                continue
            
            y = series.iloc[-lookback_k-1:].values
            # 增加对y中NaN值的检查
            if np.isnan(y).any():
                slopes[stock_code] = np.nan
                continue
            slope, _, _, _, _ = linregress(x_range, y)
            denominator = series.iloc[-lookback_k-1]
            h_t_k = slope / (denominator + self.epsilon) if denominator != 0 else 0
            slopes[stock_code] = np.tanh(a_param * h_t_k)
        return pd.Series(slopes)
    def _calc_old_i(self, adx_period=14, adx_threshold=20.0, b_param=0.075):
        move_up = self.high.diff()
        move_down = -self.low.diff()
        plus_dm = pd.DataFrame(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=self.high.index, columns=self.high.columns)
        minus_dm = pd.DataFrame(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=self.low.index, columns=self.low.columns)
        
        tr = self._calculate_tr() # 调用统一的TR计算函数
        alpha = 1 / adx_period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + self.epsilon))
        di_sum = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / di_sum.replace(0, np.inf))
        adx = dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        
        raw_i = np.tanh(b_param * (adx - adx_threshold))
        return raw_i.clip(lower=0)
    def _calc_old_m(self):
        d_t = self._calc_old_d()
        i_t = self._calc_old_i()
        return d_t * i_t
    
    def _calc_avg_amount_5d(self, period=5):
        """计算N日平均成交额"""
        return self.amount.rolling(window=period).mean().iloc[-1]
    def _calc_td_count(self, lookback=4):
        """
        计算“神奇N转”的连续计数器。
        正值表示上涨计数，负值表示下跌计数。
        当趋势中断时，计数器归零。
        :param lookback: int, 比较的周期，默认为4，即TD Sequential的经典设置。
        :return: pd.Series, 因子值序列。
        """
        close = self.df['close']
        
        # 1. 定义基础条件
        is_up_streak = (close > close.shift(lookback))
        is_down_streak = (close < close.shift(lookback))
        # 2. 计算连续计数
        # 使用 groupby 和 cumsum 的技巧来计算连续满足条件的次数
        # 当条件从 True 变为 False 或反之时，(condition != condition.shift()).cumsum() 会产生一个新的组号
        
        # 计算上涨连续计数
        up_groups = (is_up_streak != is_up_streak.shift()).cumsum()
        up_counts = is_up_streak.groupby(up_groups).cumsum()
        # 只保留上涨期间的计数，其他时间为0
        up_counts[~is_up_streak] = 0
        # 计算下跌连续计数
        down_groups = (is_down_streak != is_down_streak.shift()).cumsum()
        down_counts = is_down_streak.groupby(down_groups).cumsum()
        # 只保留下跌期间的计数，其他时间为0
        down_counts[~is_down_streak] = 0
        # 3. 合并为最终因子
        # 下跌计数为负，上涨计数为正
        td_count_factor = up_counts - down_counts
        
        return td_count_factor.iloc[-1]
# ==============================================================================
#  主服务 (Main Service)
# ==============================================================================
class StockValueService:
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
        if not self.MODEL_FILE.exists() or not self.CONFIG_FILE.exists():
            logger.error("个股评分模型或配置文件不存在。")
            return
        try:
            self._model = joblib.load(self.MODEL_FILE)
            with open(self.CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            self._feature_names = self._config['feature_names']
            logger.info("成功加载个股评分模型及配置。")
            self._dependencies_loaded = True
        except Exception as e:
            logger.error(f"加载个股评分模型依赖时出错: {e}", exc_info=True)

    def get_all_stock_scores(self, stock_pool: list, trade_date, m_value_factors: float, preloaded_panels: dict = None) -> pd.Series:
        if not self._dependencies_loaded:
            logger.warning("模型未加载，返回空评分列表。")
            return pd.Series(dtype=float)
        panel_data = {}
        # --- [关键修正] 开始 ---
        if preloaded_panels:
            logger.debug("StockValueService 检测到预加载面板数据，直接使用。")
            # 直接使用预加载的数据，但要确保列名是 'amount'
            panel_data = preloaded_panels.copy()
            if 'turnover' in panel_data and 'amount' not in panel_data:
                panel_data['amount'] = panel_data.pop('turnover')
        else:
            logger.info("未提供预加载面板，StockValueService 将从数据库加载数据...")
            # 1. 高效加载数据 (回退逻辑)
            start_date = trade_date - timedelta(days=FACTOR_LOOKBACK_BUFFER * 2)
            quotes_qs = DailyQuotes.objects.filter(
                stock_code_id__in=stock_pool,
                trade_date__gte=start_date,
                trade_date__lte=trade_date
            ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close')
            
            if not quotes_qs:
                logger.warning("在指定日期范围内未找到任何股票行情数据。")
                return pd.Series(dtype=float)
            all_quotes_df = pd.DataFrame.from_records(quotes_qs)
            
            # 2. 预处理：价格复权和列名统一
            logger.info("正在进行价格后复权处理...")
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']
            for col in numeric_cols:
                all_quotes_df[col] = pd.to_numeric(all_quotes_df[col], errors='coerce')
            
            all_quotes_df['adj_factor'] = all_quotes_df['hfq_close'] / (all_quotes_df['close'] + 1e-9)
            
            all_quotes_df['open'] = all_quotes_df['open'] * all_quotes_df['adj_factor']
            all_quotes_df['high'] = all_quotes_df['high'] * all_quotes_df['adj_factor']
            all_quotes_df['low'] = all_quotes_df['low'] * all_quotes_df['adj_factor']
            all_quotes_df['close'] = all_quotes_df['hfq_close']
            all_quotes_df.rename(columns={'turnover': 'amount'}, inplace=True)
            # 3. 构建面板数据
            logger.info("构建面板数据...")
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                panel = all_quotes_df.pivot(index='trade_date', columns='stock_code_id', values=col)
                full_date_range = pd.date_range(start=panel.index.min(), end=panel.index.max(), freq='B')
                panel = panel.reindex(full_date_range).ffill()
                panel_data[col] = panel
        # --- [关键修正] 结束 ---
        # 4. 使用向量化引擎计算所有特征
        logger.info("启动向量化因子计算引擎...")
        features_to_calc = [
            f for f in self._feature_names 
            if f not in ['market_m_value', 'm_value_lag1', 'm_value_diff1', 'm_value_ma5','avg_amount_5d''dynamic_TD_COUNT']
        ]
        logger.info("开始进行股票筛选（ST、低流动性）...")
        # 筛选条件1：过滤低流动性股票
        # 定义流动性阈值，一亿元
        LIQUIDITY_THRESHOLD = 1e8 
        # 取最近3个交易日的成交额数据
        try:
            last_3_days_amount = panel_data['amount'].iloc[-3:]
            # 检查成交额是否均小于阈值
            is_illiquid = (last_3_days_amount < LIQUIDITY_THRESHOLD).all()
            # 获取不活跃股列表
            low_liquidity_stocks = is_illiquid[is_illiquid].index.tolist()
        except IndexError:
            logger.warning("数据不足3天，无法进行流动性筛选。")
            low_liquidity_stocks = []
        logger.info(f"发现 {len(low_liquidity_stocks)} 只低流动性股票将被剔除。")
        # 筛选条件2：过滤ST类股票
        # 注意：这里需要一个获取ST股票列表的方法。
        # 假设我们有一个从数据库或其他地方获取当日ST股票列表的函数。
        # 你需要根据你的项目结构自行实现 get_st_stocks(trade_date)
        # 下面是一个示例实现，假设ST股名称中包含'ST'
        # from basic_info.models import StockInfo # 假设你有这样一个模型
        # st_stocks = list(StockInfo.objects.filter(name__contains='ST').values_list('stock_code_id', flat=True))
        # 为了演示，我们先假设一个空的列表，你需要替换它
        st_stocks = []  # <--- 重要：请替换为真实的ST股票列表获取逻辑
        logger.info(f"发现 {len(st_stocks)} 只ST股票将被剔除。")
        # 合并需要剔除的股票列表
        stocks_to_remove = set(low_liquidity_stocks) | set(st_stocks)
        # 如果有需要剔除的股票，则在 panel_data 中剔除它们
        if stocks_to_remove:
            original_stock_count = len(panel_data['close'].columns)
            # 从每个数据面板中删除这些股票的列
            for key in panel_data:
                panel_data[key].drop(columns=list(stocks_to_remove), inplace=True, errors='ignore')
            
            remaining_stock_count = len(panel_data['close'].columns)
            logger.info(f"筛选完成: 共剔除 {original_stock_count - remaining_stock_count} 只股票。剩余 {remaining_stock_count} 只。")
        engine = VectorizedFactorEngine(panel_data, features_to_calc)
        features_df = engine.run()

        # 4. 准备模型输入
        logger.info("准备模型输入并进行预测...")
        for factor_name, factor_value in m_value_factors.items():
            if factor_name in self._feature_names:
                features_df[factor_name] = factor_value
        features_df.dropna(inplace=True)
        
        if features_df.empty:
            logger.warning("所有股票在特征计算后都因NaN被剔除。")
            return pd.Series(dtype=float)

        # 保证特征顺序和数据类型
        model_input = features_df[self._feature_names].astype(float)
        
        # 5. 一次性预测所有股票
        scores = self._model.predict(model_input)
        
        # 6. 组装成最终的Series
        final_scores = pd.Series(scores, index=model_input.index)
        
        logger.info(f"成功为 {len(final_scores)} 只股票计算了模型评分。")
        return final_scores
