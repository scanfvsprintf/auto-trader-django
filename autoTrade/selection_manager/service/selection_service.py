# selection_manager/service/selection_service.py

import logging
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import linregress

from django.db import transaction
from django.utils import timezone

from common.models.stock_info import StockInfo
from common.models.daily_quotes import DailyQuotes
from common.models.system_log import SystemLog
from common.models.factor_definitions import FactorDefinitions
from common.models.daily_factor_values import DailyFactorValues
from common.models.strategy_parameters import StrategyParameters
from common.models.daily_trading_plan import DailyTradingPlan



# region: 全局配置
MODULE_NAME = '日终选股与预案生成'
logger = logging.getLogger(MODULE_NAME)
# endregion
MARKET_INDICATOR_CODE = '_MARKET_REGIME_INDICATOR_'

class SelectionService:
    """
    T-1日收盘后运行的选股与预案生成服务。
    
    执行流程:
    1. 加载策略参数。
    2. 对全市场股票进行初步筛选。
    3. 加载所需时间窗口内的所有行情数据，构建面板数据(Panel Data)。
    4. 计算所有因子原始值。
    5. 对因子值进行标准化处理。
    6. 计算综合得分并排序。
    7. 为Top N股票生成交易预案（MIOP/MAOP）。
    8. 将所有结果持久化到数据库。
    """

    def __init__(self, trade_date: date, mode: str = 'realtime', one_strategy: str = None, preloaded_panels: dict = None):
        """
        初始化选股服务。

        :param trade_date: T-1日，即执行计算的当天日期。
        :param mode: 运行模式, 'realtime' 或 'backtest'。
        """
        if mode not in ['realtime', 'backtest']:
            raise ValueError("模式(mode)必须是 'realtime' 或 'backtest'")

        self.trade_date = trade_date
        self.mode = mode
        self.params = {}
        self.factor_defs = {}
        
        # 用于存储面板数据
        self.panel_open = None
        self.panel_high = None
        self.panel_low = None
        self.panel_close = None
        self.panel_volume = None
        self.panel_turnover = None
        self.panel_hfq_close = None

        logger.debug(f"--- SelectionService 初始化 ---")
        logger.debug(f"交易日期 (T-1): {self.trade_date}")
        logger.debug(f"运行模式: {self.mode}")

    # region: --- 1. 主流程与入口方法 ---

    @staticmethod
    def initialize_strategy():
        """
        初始化策略所需的因子定义和参数到数据库。
        这是一个幂等操作，可以重复运行。
        """
        logger.debug("开始初始化策略：铺底因子定义和策略参数...")

        # 1. 定义所有因子
        factors_to_define = [
            {'factor_code': 'MA20_SLOPE', 'factor_name': '20日均线斜率', 'direction': 'positive', 'description': 'x1: 过去20个交易日MA20值的线性回归斜率'},
            {'factor_code': 'MA_ALIGNMENT', 'factor_name': '均线排列评分', 'direction': 'positive', 'description': 'x2: 收盘价/MA5/MA10/MA20多头排列评分(0-3)'},
            {'factor_code': 'ADX_TREND', 'factor_name': 'ADX趋势强度', 'direction': 'positive', 'description': 'x3: ADX > 25 且 +DI > -DI 时的ADX值'},
            {'factor_code': 'ROC10', 'factor_name': '10日价格变化率', 'direction': 'positive', 'description': 'x4: (Close_t / Close_t-10) - 1'},
            {'factor_code': 'VOL_BREAKOUT', 'factor_name': '成交量突破', 'direction': 'positive', 'description': 'x5: 最近5日均量 / 最近60日均量'},
            {'factor_code': 'NEW_HIGH_MOMENTUM', 'factor_name': '新高动能', 'direction': 'positive', 'description': 'x6: 当前收盘价 / 过去60日最高价'},
            {'factor_code': 'VOLATILITY20', 'factor_name': '20日波动率', 'direction': 'negative', 'description': 'x7: 最近20日复权收益率标准差'},
            {'factor_code': 'LIQUIDITY20', 'factor_name': '20日流动性', 'direction': 'positive', 'description': 'x8: 最近20日日均成交额'},
        ]
        with transaction.atomic():
            for factor_data in factors_to_define:
                FactorDefinitions.objects.update_or_create(
                    factor_code=factor_data['factor_code'],
                    defaults=factor_data
                )
        logger.debug(f"成功初始化/更新 {len(factors_to_define)} 个因子定义。")

        # 2. 定义所有参数 (权重、系数、计算周期)
        # !!! 要求4: 给我留一个字典当口子，后面我想好了值再填进去 !!!
        # 这里就是那个口子，您可以随时修改这些默认值
        parameters_to_define = {
            # 维度权重 (和必须为1)
            'w_trend': {'value': Decimal('0.4'), 'group': 'WEIGHTS', 'desc': '趋势维度权重'},
            'w_momentum': {'value': Decimal('0.4'), 'group': 'WEIGHTS', 'desc': '动能维度权重'},
            'w_quality': {'value': Decimal('0.2'), 'group': 'WEIGHTS', 'desc': '质量/风控维度权重'},
            # 因子权重
            'k1': {'value': Decimal('0.4'), 'group': 'TREND_FACTORS', 'desc': 'x1: MA20斜率权重'},
            'k2': {'value': Decimal('0.3'), 'group': 'TREND_FACTORS', 'desc': 'x2: 均线排列权重'},
            'k3': {'value': Decimal('0.3'), 'group': 'TREND_FACTORS', 'desc': 'x3: ADX趋势强度权重'},
            'k4': {'value': Decimal('0.4'), 'group': 'MOMENTUM_FACTORS', 'desc': 'x4: ROC10权重'},
            'k5': {'value': Decimal('0.3'), 'group': 'MOMENTUM_FACTORS', 'desc': 'x5: 成交量突破权重'},
            'k6': {'value': Decimal('0.3'), 'group': 'MOMENTUM_FACTORS', 'desc': 'x6: 新高动能权重'},
            'k7': {'value': Decimal('0.5'), 'group': 'QUALITY_FACTORS', 'desc': 'x7: 波动率权重'},
            'k8': {'value': Decimal('0.5'), 'group': 'QUALITY_FACTORS', 'desc': 'x8: 流动性权重'},
            # 交易预案参数
            'k_drop': {'value': Decimal('0.3'), 'group': 'PLAN_PARAMS', 'desc': 'MIOP低开容忍系数'},
            'k_gap': {'value': Decimal('0.5'), 'group': 'PLAN_PARAMS', 'desc': 'MAOP高开容忍系数'},
            # 计算周期参数
            'lookback_new_stock': {'value': Decimal('60'), 'group': 'LOOKBACKS', 'desc': '次新股定义天数'},
            'lookback_liquidity': {'value': Decimal('20'), 'group': 'LOOKBACKS', 'desc': '流动性计算周期'},
            'lookback_ma_slow': {'value': Decimal('60'), 'group': 'LOOKBACKS', 'desc': '慢速均线周期(如成交量)'},
            'lookback_ma_fast': {'value': Decimal('5'), 'group': 'LOOKBACKS', 'desc': '快速均线周期(如成交量)'},
            'lookback_ma20': {'value': Decimal('20'), 'group': 'LOOKBACKS', 'desc': 'MA20周期'},
            'lookback_ma10': {'value': Decimal('10'), 'group': 'LOOKBACKS', 'desc': 'MA10周期'},
            'lookback_ma5': {'value': Decimal('5'), 'group': 'LOOKBACKS', 'desc': 'MA5周期'},
            'lookback_roc': {'value': Decimal('10'), 'group': 'LOOKBACKS', 'desc': 'ROC计算周期'},
            'lookback_new_high': {'value': Decimal('60'), 'group': 'LOOKBACKS', 'desc': '新高动能计算周期'},
            'lookback_volatility': {'value': Decimal('20'), 'group': 'LOOKBACKS', 'desc': '波动率计算周期'},
            'lookback_atr': {'value': Decimal('14'), 'group': 'LOOKBACKS', 'desc': 'ATR计算周期'},
            'lookback_adx': {'value': Decimal('14'), 'group': 'LOOKBACKS', 'desc': 'ADX计算周期'},
            # 其他参数
            'param_min_liquidity': {'value': Decimal('100000000'), 'group': 'FILTERS', 'desc': '最低日均成交额(元)'},
            'param_top_n': {'value': Decimal('10'), 'group': 'SELECTION', 'desc': '最终选取股票数量'},
            'param_adx_threshold': {'value': Decimal('25'), 'group': 'THRESHOLDS', 'desc': 'ADX趋势形成阈值'},
        }
        with transaction.atomic():
            for name, data in parameters_to_define.items():
                StrategyParameters.objects.update_or_create(
                    param_name=name,
                    defaults={
                        'param_value': data['value'],
                        'group_name': data['group'],
                        'description': data['desc']
                    }
                )
        logger.debug(f"成功初始化/更新 {len(parameters_to_define)} 个策略参数。")
        logger.debug("策略初始化完成。")

    def run_selection(self):
        """
        一键启动全流程的入口方法。
        """
        # 关闭 pandas-ta 的冗余日志
        ta.Imports["verbose"] = False
        self._log_to_db('INFO', f"选股流程启动。模式: {self.mode}, 日期: {self.trade_date}")
        try:
            # 步骤 1: 加载所有配置参数
            self._load_parameters_and_defs()

            # 步骤 2: 初步筛选股票池
            initial_stock_pool = self._initial_screening()
            if not initial_stock_pool:
                logger.warning("初步筛选后无符合条件的股票，流程终止。")
                self._log_to_db('WARNING', "初步筛选后无符合条件的股票，流程终止。")
                return

            # 步骤 3: 加载行情数据并构建面板
            self._load_market_data(initial_stock_pool)

            # 步骤 4: 计算所有因子原始值
            raw_factors_df = self._calculate_all_factors()

            # 步骤 5: 标准化因子
            norm_scores_df = self._standardize_factors(raw_factors_df)

            # 步骤 6: 计算综合得分
            final_scores = self._calculate_composite_score(norm_scores_df)
            
            # 步骤 7: 生成交易预案
            trading_plan = self._generate_trading_plan(final_scores)
            if trading_plan.empty:
                logger.warning("最终未生成任何交易预案。")
                self._log_to_db('WARNING', "最终未生成任何交易预案。")
                return

            # 步骤 8: 保存所有结果到数据库
            self._save_results(raw_factors_df, norm_scores_df, trading_plan)

            # 模式特定逻辑的口子
            if self.mode == 'backtest':
                logger.debug("回测模式特定逻辑处理... (当前无)")
            elif self.mode == 'realtime':
                logger.debug("实时模式特定逻辑处理... (当前无)")

            success_msg = f"选股流程成功完成。生成 {len(trading_plan)} 条交易预案。"
            logger.info(success_msg)
            self._log_to_db('INFO', success_msg)

        except Exception as e:
            error_msg = f"选股流程发生严重错误: {e}"
            logger.critical(error_msg, exc_info=True)
            self._log_to_db('CRITICAL', error_msg)
            # 如果在事务中，需要确保事务回滚，但Django的请求/响应周期或事务装饰器通常会处理这个
            raise

    # endregion

    # region: --- 2. 内部辅助方法 ---

    def _log_to_db(self, level, message):
        """辅助方法：将日志写入数据库"""
        SystemLog.objects.create(
            log_level=level,
            module_name=MODULE_NAME,
            message=message
        )

    def _load_parameters_and_defs(self):
        """从数据库加载所有策略参数和因子定义到内存"""
        logger.debug("加载策略参数和因子定义...")
        
        # 加载参数
        params_qs = StrategyParameters.objects.all()
        # --- 修改点在这里 ---
        # 在加载时，直接将 Decimal 转换为 float，用于后续的科学计算
        # 对于需要整数的参数，单独处理
        self.params = {}
        for p in params_qs:
            if p.param_name.startswith('lookback_') or p.param_name in ['param_top_n', 'param_adx_threshold']:
                self.params[p.param_name] = int(p.param_value)
            else:
                self.params[p.param_name] = float(p.param_value)
        
        # 加载因子定义
        defs_qs = FactorDefinitions.objects.filter(is_active=True)
        self.factor_defs = {f.factor_code: {'direction': f.direction} for f in defs_qs}
        
        logger.debug(f"加载了 {len(self.params)} 个参数和 {len(self.factor_defs)} 个启用的因子定义。")

    def _get_market_trade_dates(self, lookback_period: int) -> list[date]:
        """获取截至T-1日的N个市场交易日历"""
        trade_dates = list(
            DailyQuotes.objects
            .filter(trade_date__lte=self.trade_date)
            .values_list('trade_date', flat=True)
            .distinct()
            .order_by('-trade_date')[:lookback_period]
        )
        trade_dates.reverse()
        return trade_dates

    def _initial_screening(self) -> list[str]:
        """执行初步筛选，返回符合条件的股票代码列表"""
        logger.debug("开始执行初步筛选...")
        
        # 1. 剔除ST股
        all_stocks = StockInfo.objects.filter(status=StockInfo.StatusChoices.LISTING)
        non_st_stocks = all_stocks.exclude(stock_code__contains='.688').exclude(stock_name__startswith='ST').exclude(stock_name__startswith='*ST')
        
        # 2. 剔除次新股
        min_listing_date = self.trade_date - timedelta(days=self.params['lookback_new_stock'])
        non_new_stocks = non_st_stocks.filter(listing_date__lt=min_listing_date)
        
        stock_pool_codes = list(non_new_stocks.values_list('stock_code', flat=True))
        logger.info(f"剔除ST和次新股后，剩余 {len(stock_pool_codes)} 只股票。")

        # 3. 剔除低流动性股
        lookback_days = self.params['lookback_liquidity']
        start_date = self.trade_date - timedelta(days=lookback_days * 2) 
        
        quotes = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool_codes,
            trade_date__lte=self.trade_date,
            trade_date__gte=start_date
        ).values('stock_code_id', 'trade_date', 'turnover')

        if not quotes:
            logger.warning("在流动性筛选期间未找到任何行情数据。")
            return []

        quotes_df = pd.DataFrame.from_records(quotes)
        
        # 获取最近 lookback_days 个交易日
        recent_trade_dates = sorted(quotes_df['trade_date'].unique())[-lookback_days:]
        quotes_df = quotes_df[quotes_df['trade_date'].isin(recent_trade_dates)]

        avg_turnover = quotes_df.groupby('stock_code_id')['turnover'].mean()
        liquid_stocks = avg_turnover[avg_turnover >= self.params['param_min_liquidity']]
        
        final_stock_pool = list(liquid_stocks.index)
        logger.info(f"剔除低流动性股后，最终剩余 {len(final_stock_pool)} 只股票进入精选池。")
        
        return final_stock_pool

    def _load_market_data(self, stock_pool: list[str]):
        """加载所有需要的数据并构建面板"""
        # 确定最长的回溯期
        max_lookback = max(
            self.params['lookback_ma_slow'],
            self.params['lookback_new_high'],
            self.params['lookback_adx'] + 50 # ADX需要更长的数据来稳定
        )
        logger.debug(f"确定最大数据回溯期为 {max_lookback} 个交易日。")

        trade_dates = self._get_market_trade_dates(max_lookback)
        if not trade_dates:
            raise ValueError("无法获取市场交易日历，数据库可能为空。")
        
        logger.debug(f"正在加载 {len(stock_pool)} 只股票在 {len(trade_dates)} 个交易日内的行情数据...")
        
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool,
            trade_date__in=trade_dates
        ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close')
        
        if not quotes_qs:
            raise ValueError("在指定日期范围内未找到任何股票的行情数据。")

        df = pd.DataFrame.from_records(quotes_qs)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 构建面板数据，停牌的股票会自动填充NaN
        logger.debug("正在构建面板数据(Panel Data)...")
        # 1. 从原始DataFrame中确定所有面板的“理想形状”
        # 这是最全的日期和股票列表，是我们的“真理之源”
        master_index_raw = df['trade_date'].unique()
        master_columns_raw = df['stock_code_id'].unique()
        master_index = np.sort(master_index_raw)
        master_columns = np.sort(master_columns_raw)
        
        
        
        logger.debug(f"已确定基准形状: {len(master_index)} 个日期 x {len(master_columns)} 只股票。")
 
        # 2. 使用 set_index + unstack 高效地创建各个面板
        # 这一步创建的面板可能形状不一，因为个别股票在某些天可能没有某个值
        base_df = df.set_index(['trade_date', 'stock_code_id'])
        
        panel_open_raw = base_df['open'].unstack()
        panel_high_raw = base_df['high'].unstack()
        panel_low_raw = base_df['low'].unstack()
        panel_close_raw = base_df['close'].unstack()
        panel_volume_raw = base_df['volume'].unstack()
        panel_turnover_raw = base_df['turnover'].unstack()
        panel_hfq_close_raw = base_df['hfq_close'].unstack()
 
        # 3. 将所有原始面板 reindex 到我们预先定义的“理想形状”
        # 这样可以确保所有面板的维度完全一致，缺失值会被正确地填充为NaN
        self.panel_open = panel_open_raw.reindex(index=master_index, columns=master_columns)
        self.panel_high = panel_high_raw.reindex(index=master_index, columns=master_columns)
        self.panel_low = panel_low_raw.reindex(index=master_index, columns=master_columns)
        self.panel_close = panel_close_raw.reindex(index=master_index, columns=master_columns)
        self.panel_volume = panel_volume_raw.reindex(index=master_index, columns=master_columns)
        self.panel_turnover = panel_turnover_raw.reindex(index=master_index, columns=master_columns)
        self.panel_hfq_close = panel_hfq_close_raw.reindex(index=master_index, columns=master_columns)
        self.panel_open = self.panel_open.astype(float)
        self.panel_high = self.panel_high.astype(float)
        self.panel_low = self.panel_low.astype(float)
        self.panel_close = self.panel_close.astype(float)
        self.panel_volume = self.panel_volume.astype(float)
        self.panel_turnover = self.panel_turnover.astype(float)
        self.panel_hfq_close = self.panel_hfq_close.astype(float)
        logger.debug("面板数据构建完成。")

    # endregion

    # region: --- 3. 因子计算 (模块化) ---

    def _calculate_all_factors(self) -> pd.DataFrame:
        """
        调度所有因子计算方法，并将结果合并到一个DataFrame中。
        返回的DataFrame: index=stock_code, columns=factor_codes
        """
        logger.debug("开始计算所有因子...")
        
        factor_calculators = {
            'MA20_SLOPE': self._calc_factor_x1_ma20_slope,
            'MA_ALIGNMENT': self._calc_factor_x2_ma_alignment,
            'ADX_TREND': self._calc_factor_x3_adx_trend,
            'ROC10': self._calc_factor_x4_roc,
            'VOL_BREAKOUT': self._calc_factor_x5_vol_breakout,
            'NEW_HIGH_MOMENTUM': self._calc_factor_x6_new_high_momentum,
            'VOLATILITY20': self._calc_factor_x7_volatility,
            'LIQUIDITY20': self._calc_factor_x8_liquidity,
        }
        
        all_factors = {}
        for code, func in factor_calculators.items():
            if code in self.factor_defs:
                logger.debug(f"  - 计算因子: {code}")
                # 每个因子计算方法返回一个以stock_code为索引的Series
                all_factors[code] = func()
        
        # 合并所有因子Series为一个DataFrame
        raw_factors_df = pd.DataFrame(all_factors)
        
        # 剔除任何一个因子值为NaN的股票
        original_count = len(raw_factors_df)
        raw_factors_df.dropna(inplace=True)
        final_count = len(raw_factors_df)
        logger.info(f"因子计算完成。因数据不足(NaN)剔除了 {original_count - final_count} 只股票。剩余 {final_count} 只。")
        
        return raw_factors_df

    # --- 每个因子的独立计算方法 ---
    # 传参: 无 (通过self访问面板数据和参数)
    # 返回: pd.Series (index=stock_code, value=因子原始值)

    def _calc_factor_x1_ma20_slope(self) -> pd.Series:
        ma_period = self.params['lookback_ma20']
        panel_close_float = self.panel_hfq_close.astype(float)
        ma20 = panel_close_float.apply(
            lambda col: ta.sma(close=col, length=ma_period)
        )
        
        # 对每只股票，取最后ma_period个MA值进行线性回归
        def get_slope(series):
            y = series.dropna()
            if len(y) < 2:
                return np.nan
            y_values = y.values
            x_values = np.arange(len(y_values))
            # 增加一个额外的健壮性检查，防止y_values中所有值都相同导致回归失败
            if np.all(y_values == y_values[0]):
                return 0.0 # 如果所有值都一样，斜率为0
            slope, _, _, _, _ = linregress(x_values, y_values)
            return slope

        # 只取最后ma_period行数据进行计算以提高效率
        slopes = ma20.iloc[-ma_period:].apply(get_slope, axis=0)
        return slopes

    def _calc_factor_x2_ma_alignment(self) -> pd.Series:
        ma5 = self.panel_hfq_close.apply(
            lambda col: ta.sma(close=col, length=self.params['lookback_ma5'])
        )
        ma10 = self.panel_hfq_close.apply(
            lambda col: ta.sma(close=col, length=self.params['lookback_ma10'])
        )
        ma20 = self.panel_hfq_close.apply(
            lambda col: ta.sma(close=col, length=self.params['lookback_ma20'])
        )
        
        last_close = self.panel_hfq_close.iloc[-1]
        last_ma5 = ma5.iloc[-1]
        last_ma10 = ma10.iloc[-1]
        last_ma20 = ma20.iloc[-1]
        
        score = (
            (last_close > last_ma5).astype(int) +
            (last_ma5 > last_ma10).astype(int) +
            (last_ma10 > last_ma20).astype(int)
        )
        return score

    def _calc_factor_x3_adx_trend(self) -> pd.Series:
        # 强制转换数据类型为 float
        high = self.panel_high.astype(float)
        low = self.panel_low.astype(float)
        close = self.panel_close.astype(float)
        
        adx_period = self.params['lookback_adx']
        threshold = self.params['param_adx_threshold']
 
        # 定义一个函数，用于对单只股票（一个Series）计算ADX
        def get_adx_trend(stock_code):
            # 从面板中提取单只股票的数据
            stock_high = high[stock_code].dropna()
            stock_low = low[stock_code].dropna()
            stock_close = close[stock_code].dropna()
 
            # 确保数据对齐
            common_index = stock_high.index.intersection(stock_low.index).intersection(stock_close.index)
            if len(common_index) < adx_period * 2: # ADX需要更长的数据来初始化
                return np.nan
 
            stock_high = stock_high.loc[common_index]
            stock_low = stock_low.loc[common_index]
            stock_close = stock_close.loc[common_index]
 
            # 使用 pandas_ta 计算单只股票的ADX
            adx_result = ta.adx(high=stock_high, low=stock_low, close=stock_close, length=adx_period)
 
            # 检查结果是否有效
            if adx_result is None or adx_result.empty:
                return np.nan
 
            last_row = adx_result.iloc[-1]
            last_adx = last_row.get(f'ADX_{adx_period}')
            last_dmp = last_row.get(f'DMP_{adx_period}')
            last_dmn = last_row.get(f'DMN_{adx_period}')
 
            if pd.isna(last_adx) or pd.isna(last_dmp) or pd.isna(last_dmn):
                return np.nan
 
            if last_adx > threshold and last_dmp > last_dmn:
                return last_adx
            else:
                return 0.0
 
        # 对面板中的每一只股票应用该函数
        results = {stock_code: get_adx_trend(stock_code) for stock_code in self.panel_close.columns}
        
        return pd.Series(results)

    def _calc_factor_x4_roc(self) -> pd.Series:
        roc_panel = self.panel_hfq_close.apply(
        lambda col: ta.roc(close=col, length=self.params['lookback_roc'])
        )
        return roc_panel.iloc[-1]

    def _calc_factor_x5_vol_breakout(self) -> pd.Series:
        vol5 = self.panel_volume.rolling(window=self.params['lookback_ma_fast']).mean()
        vol60 = self.panel_volume.rolling(window=self.params['lookback_ma_slow']).mean()
        ratio = vol5.iloc[-1] / vol60.iloc[-1]
        return ratio.replace([np.inf, -np.inf], np.nan) # 处理分母为0的情况

    def _calc_factor_x6_new_high_momentum(self) -> pd.Series:
        high60 = self.panel_hfq_close.rolling(window=self.params['lookback_new_high']).max()
        ratio = self.panel_hfq_close.iloc[-1] / high60.iloc[-1]
        return ratio

    def _calc_factor_x7_volatility(self) -> pd.Series:
        returns = self.panel_hfq_close.pct_change(fill_method=None)
        volatility = returns.rolling(window=self.params['lookback_volatility']).std()
        return volatility.iloc[-1]

    def _calc_factor_x8_liquidity(self) -> pd.Series:
        liquidity = self.panel_turnover.rolling(window=self.params['lookback_liquidity']).mean()
        return liquidity.iloc[-1]

    # endregion

    # region: --- 4. 评分、预案生成与保存 ---

    def _standardize_factors(self, raw_factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        对原始因子值进行标准化处理 (norm函数)。
        """
        logger.debug("开始对因子值进行标准化...")
        norm_scores_df = pd.DataFrame(index=raw_factors_df.index)
        
        for factor_code, series in raw_factors_df.items():
            direction = self.factor_defs[factor_code]['direction']
            
            p1 = series.quantile(0.01)
            p99 = series.quantile(0.99)
            
            
            epsilon = 1e-9  # 定义一个极小值
            if (p99 - p1) < epsilon:
                norm_scores_df[factor_code] = 0
                continue
            
            # Winsorization (缩尾处理)
            x_prime = series.clip(p1, p99)
            
            # 线性映射
            if direction == 'positive':
                score = ((x_prime - p1) / (p99 - p1)) * 200 - 100
            else: # negative
                score = ((p99 - x_prime) / (p99 - p1)) * 200 - 100
            
            norm_scores_df[factor_code] = score
            
        logger.debug("因子标准化完成。")
        return norm_scores_df

    def _calculate_composite_score(self, norm_scores_df: pd.DataFrame) -> pd.Series:
        """
        计算f(x)综合得分。
        """
        logger.debug("开始计算综合得分 f(x)...")
        
        # 趋势维度
        score_trend = (
            norm_scores_df['MA20_SLOPE'] * self.params['k1'] +
            norm_scores_df['MA_ALIGNMENT'] * self.params['k2'] +
            norm_scores_df['ADX_TREND'] * self.params['k3']
        )
        
        # 动能维度
        score_momentum = (
            norm_scores_df['ROC10'] * self.params['k4'] +
            norm_scores_df['VOL_BREAKOUT'] * self.params['k5'] +
            norm_scores_df['NEW_HIGH_MOMENTUM'] * self.params['k6']
        )
        
        # 质量/风控维度
        score_quality = (
            norm_scores_df['VOLATILITY20'] * self.params['k7'] +
            norm_scores_df['LIQUIDITY20'] * self.params['k8']
        )
        
        # 总分
        final_score = (
            score_trend * self.params['w_trend'] +
            score_momentum * self.params['w_momentum'] +
            score_quality * self.params['w_quality']
        )
        
        logger.debug("综合得分计算完成。")
        return final_score.sort_values(ascending=False)

    def _generate_trading_plan(self, final_scores: pd.Series) -> pd.DataFrame:
        top_n = self.params['param_top_n']
        top_stocks = final_scores.head(top_n)
        
        logger.debug(f"开始为Top {top_n} 股票生成交易预案...")
        
        if top_stocks.empty:
            return pd.DataFrame()
 
        top_stock_codes = top_stocks.index.tolist()
        
        # --- 修改点在这里：需要把完整的ATR计算逻辑加回来 ---
        atr_period = self.params['lookback_atr']
        last_atr_values = {}
        for code in top_stock_codes:
            # 提取单只股票数据并计算ATR
            stock_high = self.panel_high[code].astype(float).dropna()
            stock_low = self.panel_low[code].astype(float).dropna()
            stock_close = self.panel_close[code].astype(float).dropna()
            
            common_index = stock_high.index.intersection(stock_low.index).intersection(stock_close.index)
            if len(common_index) < atr_period:
                last_atr_values[code] = np.nan
                continue
 
            atr_result = ta.atr(
                high=stock_high.loc[common_index],
                low=stock_low.loc[common_index],
                close=stock_close.loc[common_index],
                length=atr_period
            )
            if atr_result is not None and not atr_result.empty:
                last_atr_values[code] = atr_result.iloc[-1]
            else:
                last_atr_values[code] = np.nan
        # --- ATR计算逻辑结束 ---
 
        last_atr = pd.Series(last_atr_values)
        
        # 1. 获取所有需要的数据，并确保它们是Series，且索引为stock_code
        last_close = self.panel_close.iloc[-1]
 
        # 2. 以 top_stocks 为基准，重新索引（reindex）所有数据
        aligned_scores = top_stocks
        aligned_close = last_close.reindex(top_stock_codes)
        aligned_atr = last_atr.reindex(top_stock_codes)
 
        # 3. 在对齐后的数据上进行计算
        k_drop = self.params['k_drop']
        k_gap = self.params['k_gap']
        
        miop = aligned_close * (1 - k_drop * (aligned_atr / aligned_close))
        maop = aligned_close * (1 + k_gap * (aligned_atr / aligned_close))
 
        # 4. 组装预案
        plan_df = pd.DataFrame({
            'stock_code': top_stock_codes,
            'rank': range(1, len(top_stock_codes) + 1),
            'final_score': aligned_scores.values,
            'miop': miop.values,
            'maop': maop.values
        })
 
        # 5. (可选但推荐) 剔除因为数据不足无法生成预案的行
        plan_df.dropna(subset=['miop', 'maop'], inplace=True)
        
        logger.debug("交易预案生成完成。")
        return plan_df

    @transaction.atomic
    def _save_results(self, raw_factors_df, norm_scores_df, trading_plan_df):
        """
        将所有计算结果原子性地保存到数据库。
        """
        logger.debug("开始将结果保存到数据库...")
        
        # 1. 保存每日因子值
        logger.debug(f"  - 正在准备 {len(raw_factors_df) * len(raw_factors_df.columns)} 条因子值数据...")
        factor_values_to_create = []
        for stock_code, row in raw_factors_df.iterrows():
            for factor_code, raw_value in row.items():
                norm_score = norm_scores_df.loc[stock_code, factor_code]
                factor_values_to_create.append(
                    DailyFactorValues(
                        stock_code_id=stock_code,
                        trade_date=self.trade_date,
                        factor_code_id=factor_code,
                        raw_value=min(Decimal(str(raw_value)),Decimal(999999)),
                        norm_score=Decimal(str(norm_score))
                    )
                )
        DailyFactorValues.objects.bulk_create(factor_values_to_create, ignore_conflicts=True)
        logger.debug(f"  - 成功保存因子值。")

        # 2. 保存每日交易预案
        plan_date = self.trade_date + timedelta(days=1)
        logger.debug(f"  - 正在为 {plan_date} 保存 {len(trading_plan_df)} 条交易预案...")
        
        # 先删除当天的旧预案，以防重复运行
        DailyTradingPlan.objects.filter(plan_date=plan_date).delete()
        
        plans_to_create = []
        for _, row in trading_plan_df.iterrows():
            plans_to_create.append(
                DailyTradingPlan(
                    plan_date=plan_date,
                    stock_code_id=row['stock_code'],
                    rank=row['rank'],
                    final_score=Decimal(str(row['final_score'])),
                    miop=Decimal(str(row['miop'])).quantize(Decimal('0.01')),
                    maop=Decimal(str(row['maop'])).quantize(Decimal('0.01')),
                    status=DailyTradingPlan.StatusChoices.PENDING
                )
            )
        DailyTradingPlan.objects.bulk_create(plans_to_create)
        logger.debug(f"  - 成功保存交易预案。")

        # 3. 将最终选股结果记录到系统日志
        log_message = f"T-1日({self.trade_date})选股完成, T日({plan_date})预案如下:\n"
        log_message += trading_plan_df.to_string(index=False)
        self._log_to_db('INFO', log_message)
        
        logger.debug("所有结果已成功保存到数据库。")

    # endregion


# region: --- 示例用法 ---
# 如何在项目中使用这个服务

def setup_django_env():
    """
    独立脚本运行时，需要配置Django环境。
    在Django项目内部（如management command）调用时，则不需要此函数。
    """
    import os
    import django
    # 替换 'your_project.settings' 为你的项目设置文件
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
    django.setup()

if __name__ == '__main__':
    # --- 这是一个演示如何调用服务的示例 ---
    # 1. 首先需要设置Django环境
    # setup_django_env() # 如果是独立脚本，取消此行注释

    # 2. (可选) 第一次运行时，初始化策略定义
    print("="*50)
    print("步骤 1: 初始化策略 (如果需要)")
    SelectionService.initialize_strategy()
    print("="*50)

    # 3. 运行选股流程 (以回测模式为例)
    print("\n" + "="*50)
    print("步骤 2: 运行选股流程")
    # 假设我们想为 2023-10-26 (T日) 生成预案，那么T-1日就是 2023-10-25
    # 注意：请确保你的数据库中有 2023-10-25 及之前足够多的数据
    target_trade_date = date(2023, 10, 25) 
    
    try:
        service = SelectionService(trade_date=target_trade_date, mode='backtest')
        service.run_selection()
    except Exception as e:
        print(f"\n在为日期 {target_trade_date} 运行选股时发生错误: {e}")
    
    print("="*50)
    print("示例运行结束。")

# endregion
