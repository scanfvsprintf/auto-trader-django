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

from common.models import (
    StockInfo, DailyQuotes, SystemLog, FactorDefinitions, DailyFactorValues,
    StrategyParameters, DailyTradingPlan,IndexQuotesCsi300 
)

# region: 全局配置
MODULE_NAME = '日终选股与预案模块(动态版)'
logger = logging.getLogger(__name__)
MARKET_INDICATOR_CODE = '_MARKET_REGIME_INDICATOR_'
# endregion


class SelectionService:
    """
    T-1日收盘后运行的动态自适应选股与预案生成服务。
    
    该服务实现了 f_dynamic(x, t) 选股评分函数，能够根据市场状态 M(t)
    动态调整选股策略的维度权重，以适应不同的市场环境。
    """

    def __init__(self, trade_date: date, mode: str = 'realtime'):
        """
        初始化选股服务。

        :param trade_date: T-1日，即执行计算的当天日期。
        :param mode: 运行模式, 'realtime' 或 'backtest'。
        """
        if mode not in ['realtime', 'backtest']:
            raise ValueError("模式(mode)必须是 'realtime' 或 'backtest'")

        # --- 日期对齐逻辑 ---
        try:
            # 查询小于等于输入日期的、最新的一个交易日
            latest_trade_date_obj = DailyQuotes.objects.filter(trade_date__lte=trade_date).latest('trade_date')
            validated_trade_date = latest_trade_date_obj.trade_date
            
            if validated_trade_date != trade_date:
                logger.debug(f"输入日期 {trade_date} 不是交易日，已自动校准为最近的交易日: {validated_trade_date}")
            
            self.trade_date = validated_trade_date
        except :
            # 如果数据库中没有任何早于或等于 trade_date 的数据
            # error_msg = f"无法在数据库中找到日期 {trade_date} 或之前的任何交易日数据，服务无法初始化。"
            # logger.error(error_msg)
            # raise ValueError(error_msg)
            self.trade_date=trade_date
        self.mode = mode
        self.dynamic_params = {}
        self.dynamic_factor_defs = {}
        
        # 市场状态与动态权重
        self.market_regime_M = 0.0
        self.dynamic_weights = {}

        # 面板数据
        self.panel_open = None
        self.panel_high = None
        self.panel_low = None
        self.panel_close = None
        self.panel_volume = None
        self.panel_turnover = None
        self.panel_hfq_close = None

        logger.debug(f"--- SelectionService(动态版) 初始化 ---")
        logger.debug(f"交易日期 (T-1): {self.trade_date}")
        logger.debug(f"运行模式: {self.mode}")

    # region: --- 1. 主流程与入口方法 ---

    @staticmethod
    def initialize_strategy():
        """
        初始化动态策略所需的因子定义和参数到数据库。
        这是一个幂等操作，可以重复运行。
        """
        logger.debug("开始初始化动态策略：铺底因子定义和策略参数...")

        # 1. 确保市场指标的特殊股票代码存在
        try:
            StockInfo.objects.get_or_create(
                stock_code=MARKET_INDICATOR_CODE,
                defaults={
                    'stock_name': '市场状态指标',
                    'listing_date': date(1990, 1, 1),
                    'status': StockInfo.StatusChoices.LISTING
                }
            )
            logger.debug(f"特殊股票代码 '{MARKET_INDICATOR_CODE}' 已确认存在。")
        except Exception as e:
            logger.error(f"创建特殊股票代码 '{MARKET_INDICATOR_CODE}' 失败: {e}")
            # 这是一个关键步骤，如果失败则不应继续
            return

        # 2. 定义所有因子 (包括M(t)缓存因子和个股因子)
        factors_to_define = [
            # M(t) 缓存因子
            {'factor_code': 'dynamic_M_VALUE', 'factor_name': '动态-市场状态M(t)最终值', 'direction': 'positive'},
            {'factor_code': 'dynamic_M1_RAW', 'factor_name': '动态-M1原始值(新高占比)', 'direction': 'positive'},
            {'factor_code': 'dynamic_M2_RAW', 'factor_name': '动态-M2原始值(MA60之上占比)', 'direction': 'positive'},
            {'factor_code': 'dynamic_M3_RAW', 'factor_name': '动态-M3原始值(60日回报中位数)', 'direction': 'positive'},
            {'factor_code': 'dynamic_M4_RAW', 'factor_name': '动态-M4原始值(20日平均波动率)', 'direction': 'negative'},
            
            # 趋势动能 (MT) 维度因子
            {'factor_code': 'dynamic_MA20_SLOPE', 'factor_name': '动态-20日均线斜率', 'direction': 'positive'},
            {'factor_code': 'dynamic_MA_SCORE', 'factor_name': '动态-均线排列评分', 'direction': 'positive'},
            {'factor_code': 'dynamic_ADX_CONFIRM', 'factor_name': '动态-ADX趋势确认', 'direction': 'positive'},
            
            # 强势突破 (BO) 维度因子
            {'factor_code': 'dynamic_BREAKOUT_PWR', 'factor_name': '动态-突破强度', 'direction': 'positive'},
            {'factor_code': 'dynamic_VOLUME_SURGE', 'factor_name': '动态-成交量激增', 'direction': 'positive'},
            {'factor_code': 'dynamic_MOM_ACCEL', 'factor_name': '动态-动能加速度', 'direction': 'positive'},

            # 均值回归 (MR) 维度因子
            {'factor_code': 'dynamic_RSI_OS', 'factor_name': '动态-短期超卖(RSI14)', 'direction': 'negative'},
            {'factor_code': 'dynamic_NEG_DEV', 'factor_name': '动态-负向偏离度(vs MA60)', 'direction': 'negative'},
            {'factor_code': 'dynamic_BOLL_LB', 'factor_name': '动态-布林下轨支撑', 'direction': 'negative'},

            # 质量防御 (QD) 维度因子
            {'factor_code': 'dynamic_LOW_VOL', 'factor_name': '动态-低波动率(20日)', 'direction': 'negative'},
            {'factor_code': 'dynamic_MAX_DD', 'factor_name': '动态-最大回撤控制(60日)', 'direction': 'negative'},
            {'factor_code': 'dynamic_DOWNSIDE_RISK', 'factor_name': '动态-下行风险(60日)', 'direction': 'negative'},
        ]
        with transaction.atomic():
            for factor_data in factors_to_define:
                FactorDefinitions.objects.update_or_create(
                    factor_code=factor_data['factor_code'],
                    defaults={'factor_name': factor_data['factor_name'], 'direction': factor_data['direction'], 'is_active': True}
                )
        logger.debug(f"成功初始化/更新 {len(factors_to_define)} 个动态因子定义。")

        # 3. 定义所有参数 (使用 'dynamic_' 前缀)
        parameters_to_define = {
            # M(t) 参数
            # 新版 M(t) 参数 (基于沪深300)
            'dynamic_m_csi300_w_trend': {'value': '0.4', 'group': 'M_CSI300_WEIGHTS', 'desc': '新M(t)权重: 趋势强度'},
            'dynamic_m_csi300_w_momentum': {'value': '0.3', 'group': 'M_CSI300_WEIGHTS', 'desc': '新M(t)权重: 短期动能'},
            'dynamic_m_csi300_w_volatility': {'value': '0.2', 'group': 'M_CSI300_WEIGHTS', 'desc': '新M(t)权重: 波动水平'},
            'dynamic_m_csi300_w_turnover': {'value': '0.1', 'group': 'M_CSI300_WEIGHTS', 'desc': '新M(t)权重: 量能状态'},
            # 新版 M(t) 锚点参数 (由校准命令填充)
            'dynamic_m_csi300_anchor_trend_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '趋势强度-P10锚点'},
            'dynamic_m_csi300_anchor_trend_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '趋势强度-P50锚点'},
            'dynamic_m_csi300_anchor_trend_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '趋势强度-P90锚点'},
            'dynamic_m_csi300_anchor_momentum_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '短期动能-P10锚点'},
            'dynamic_m_csi300_anchor_momentum_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '短期动能-P50锚点'},
            'dynamic_m_csi300_anchor_momentum_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '短期动能-P90锚点'},
            'dynamic_m_csi300_anchor_volatility_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '波动水平-P10锚点'},
            'dynamic_m_csi300_anchor_volatility_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '波动水平-P50锚点'},
            'dynamic_m_csi300_anchor_volatility_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '波动水平-P90锚点'},
            'dynamic_m_csi300_anchor_turnover_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '量能状态-P10锚点'},
            'dynamic_m_csi300_anchor_turnover_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '量能状态-P50锚点'},
            'dynamic_m_csi300_anchor_turnover_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '量能状态-P90锚点'},

            # 动态权重 N_i(M(t)) 参数
            'dynamic_c_MT': {'value': '1.2', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 趋势动能'},
            'dynamic_c_BO': {'value': '0.7', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 强势突破'},
            'dynamic_c_QD': {'value': '1.1', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 质量防御'},
            'dynamic_c_MR': {'value': '1.5', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 均值回归'},
            'dynamic_sigma_MR': {'value': '0.25', 'group': 'N_PARAMS', 'desc': '均值回归策略适用范围宽度'},
            'dynamic_tau': {'value': '0.7', 'group': 'N_PARAMS', 'desc': 'Softmax温度系数(控制切换灵敏度)'},

            # 维度内部因子权重 (k_ij)
            'dynamic_k_MT1': {'value': '0.5', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-MA20斜率权重'},
            'dynamic_k_MT2': {'value': '0.3', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-均线排列权重'},
            'dynamic_k_MT3': {'value': '0.2', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-ADX确认权重'},
            'dynamic_k_BO1': {'value': '0.45', 'group': 'K_WEIGHTS_BO', 'desc': '强势突破-突破强度权重'},
            'dynamic_k_BO2': {'value': '0.35', 'group': 'K_WEIGHTS_BO', 'desc': '强势突破-成交量激增权重'},
            'dynamic_k_BO3': {'value': '0.2', 'group': 'K_WEIGHTS_BO', 'desc': '强势突破-动能加速度权重'},
            'dynamic_k_MR1': {'value': '0.3', 'group': 'K_WEIGHTS_MR', 'desc': '均值回归-RSI超卖权重'},
            'dynamic_k_MR2': {'value': '0.5', 'group': 'K_WEIGHTS_MR', 'desc': '均值回归-负向偏离度权重'},
            'dynamic_k_MR3': {'value': '0.2', 'group': 'K_WEIGHTS_MR', 'desc': '均值回归-布林下轨支撑权重'},
            'dynamic_k_QD1': {'value': '0.45', 'group': 'K_WEIGHTS_QD', 'desc': '质量防御-低波动率权重'},
            'dynamic_k_QD2': {'value': '0.35', 'group': 'K_WEIGHTS_QD', 'desc': '质量防御-最大回撤权重'},
            'dynamic_k_QD3': {'value': '0.2', 'group': 'K_WEIGHTS_QD', 'desc': '质量防御-下行风险权重'},
            
            # 交易预案参数
            # 'dynamic_k_drop': {'value': '0.3', 'group': 'PLAN_PARAMS', 'desc': 'MIOP低开容忍系数'},
            # 'dynamic_k_gap': {'value': '0.5', 'group': 'PLAN_PARAMS', 'desc': 'MAOP高开容忍系数'},
            'dynamic_miopmaop_k_gap_base_mt': {'value': '0.6', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-趋势动能-高开容忍度'},
            'dynamic_miopmaop_k_drop_base_mt': {'value': '0.5', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-趋势动能-低开容忍度'},
            'dynamic_miopmaop_k_gap_base_bo': {'value': '1.5', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-强势突破-高开容忍度'},
            'dynamic_miopmaop_k_drop_base_bo': {'value': '0.1', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-强势突破-低开容忍度'},
            'dynamic_miopmaop_k_gap_base_mr': {'value': '0.01', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-均值回归-高开容忍度'},
            'dynamic_miopmaop_k_drop_base_mr': {'value': '1.8', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-均值回归-低开容忍度'},
            'dynamic_miopmaop_k_gap_base_qd': {'value': '0.15', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-质量防御-高开容忍度'},
            'dynamic_miopmaop_k_drop_base_qd': {'value': '0.15', 'group': 'MIOPMAOP_BASE', 'desc': '动态开盘区间-质量防御-低开容忍度'},
            
            # 筛选与计算参数
            'dynamic_lookback_new_stock': {'value': '60', 'group': 'FILTERS', 'desc': '次新股定义天数(自然日)'},
            'dynamic_min_liquidity': {'value': '100000000', 'group': 'FILTERS', 'desc': '最低日均成交额(元)'},
            'dynamic_top_n': {'value': '10', 'group': 'SELECTION', 'desc': '最终选取股票数量'},
        }
        with transaction.atomic():
            for name, data in parameters_to_define.items():
                StrategyParameters.objects.update_or_create(
                    param_name=name,
                    defaults={
                        'param_value': Decimal(data['value']),
                        'group_name': data['group'],
                        'description': data['desc']
                    }
                )
        logger.debug(f"成功初始化/更新 {len(parameters_to_define)} 个动态策略参数。")
        logger.debug("动态策略初始化完成。")

    def run_selection(self):
        """
        一键启动全流程的入口方法。
        """
        ta.Imports["verbose"] = False
        self._log_to_db('INFO', f"动态选股流程启动。模式: {self.mode}, 日期: {self.trade_date}")
        try:
            self._load_dynamic_parameters_and_defs()
            initial_stock_pool = self._initial_screening()
            if not initial_stock_pool:
                self._log_to_db('WARNING', "初步筛选后无符合条件的股票，流程终止。")
                return

            # 核心动态逻辑
            self.market_regime_M = self._calculate_market_regime_M(initial_stock_pool)
            self.dynamic_weights = self._calculate_dynamic_weights(self.market_regime_M)
            
            self._load_market_data(initial_stock_pool)
            raw_factors_df = self._calculate_all_dynamic_factors()
            norm_scores_df = self._standardize_factors(raw_factors_df)
            
            dimension_scores_df = self._calculate_dimension_scores(norm_scores_df)
            final_scores = self._calculate_final_dynamic_score(dimension_scores_df, self.dynamic_weights)
            
            trading_plan = self._generate_trading_plan(final_scores,dimension_scores_df)
            if trading_plan.empty:
                self._log_to_db('WARNING', "最终未生成任何交易预案。")
                return

            self._save_results(raw_factors_df, norm_scores_df, trading_plan)

            success_msg = f"动态选股流程成功完成。M(t)={self.market_regime_M:.4f}, 生成 {len(trading_plan)} 条交易预案。"
            logger.debug(success_msg)
            self._log_to_db('INFO', success_msg)

        except Exception as e:
            error_msg = f"动态选股流程发生严重错误: {e}"
            logger.critical(error_msg, exc_info=True)
            self._log_to_db('CRITICAL', error_msg)
            raise

    # endregion

    # region: --- 2. 核心动态逻辑实现 ---

    def _calculate_market_regime_M(self, stock_pool: list[str]) -> float:
        """
        计算市场状态函数 M(t) - V2.0 沪深300基准版
        使用固定分位锚点法进行绝对标准化，并具备容错机制。
        """
        logger.debug("开始计算市场状态 M(t) [V2.0]...")
        # 1. 检查当日缓存
        try:
            cached_m = DailyFactorValues.objects.get(
                stock_code_id=MARKET_INDICATOR_CODE,
                trade_date=self.trade_date,
                factor_code_id='dynamic_M_VALUE'
            )
            m_value = float(cached_m.raw_value)
            logger.debug(f"成功从缓存中读取当日 M(t) = {m_value:.4f}")
            return m_value
        except DailyFactorValues.DoesNotExist:
            logger.debug("当日 M(t) 缓存未命中，开始计算...")
        # 2. 加载计算所需数据 (约60个交易日)
        try:
            lookback_days = 80  # 60日均线需要，给足buffer
            start_date = self.trade_date - timedelta(days=lookback_days * 2)
            
            quotes_qs = IndexQuotesCsi300.objects.filter(
                trade_date__gte=start_date,
                trade_date__lte=self.trade_date
            ).order_by('trade_date')
            
            if len(quotes_qs) < 60:
                raise ValueError("沪深300历史数据不足60天，无法计算M值。")
            
            df = pd.DataFrame.from_records(quotes_qs.values())
            df.set_index('trade_date', inplace=True)
            columns_to_convert = ['open', 'high', 'low', 'close', 'turnover_rate']
            for col in columns_to_convert:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except (ValueError, IndexQuotesCsi300.DoesNotExist) as e:
            logger.error(f"获取沪深300数据失败: {e}。将启用容错机制。")
            # 容错机制 (方案A): 沿用前一个交易日的M值
            try:
                latest_m = DailyFactorValues.objects.filter(
                    stock_code_id=MARKET_INDICATOR_CODE,
                    factor_code_id='dynamic_M_VALUE'
                ).latest('trade_date')
                logger.warning(f"数据获取失败，沿用 {latest_m.trade_date} 的M值: {latest_m.raw_value:.4f}")
                return float(latest_m.raw_value)
            except DailyFactorValues.DoesNotExist:
                logger.critical("数据库中无任何历史M值，无法回退，返回0。")
                return 0.0
        # 3. 计算当日的四个原始指标值
        # M1: 趋势强度
        ma60 = df['close'].rolling(60, min_periods=30).mean()
        m1_series = (df['close'] - ma60) / ma60
        m1_raw = m1_series.iloc[-1]
        # M2: 短期动能
        m2_series = df['close'].pct_change(20)
        m2_raw = m2_series.iloc[-1]
        
        # M3: 波动水平
        daily_return_series = df['close'].pct_change()
        m3_series = daily_return_series.rolling(20, min_periods=10).std()
        m3_raw = m3_series.iloc[-1]
        
        # M4: 量能状态
        avg_turnover_20 = df['turnover_rate'].rolling(20, min_periods=10).mean()
        avg_turnover_60 = df['turnover_rate'].rolling(60, min_periods=30).mean()
        m4_series = avg_turnover_20 / avg_turnover_60
        m4_raw = m4_series.iloc[-1]
        # 4. 使用固定锚点进行绝对标准化
        p = self.dynamic_params
        m1_norm = self._norm_absolute(m1_raw, p['dynamic_m_csi300_anchor_trend_p10'], p['dynamic_m_csi300_anchor_trend_p50'], p['dynamic_m_csi300_anchor_trend_p90'], 'positive')
        m2_norm = self._norm_absolute(m2_raw, p['dynamic_m_csi300_anchor_momentum_p10'], p['dynamic_m_csi300_anchor_momentum_p50'], p['dynamic_m_csi300_anchor_momentum_p90'], 'positive')
        m3_norm = self._norm_absolute(m3_raw, p['dynamic_m_csi300_anchor_volatility_p10'], p['dynamic_m_csi300_anchor_volatility_p50'], p['dynamic_m_csi300_anchor_volatility_p90'], 'negative')
        m4_norm = self._norm_absolute(m4_raw, p['dynamic_m_csi300_anchor_turnover_p10'], p['dynamic_m_csi300_anchor_turnover_p50'], p['dynamic_m_csi300_anchor_turnover_p90'], 'positive')
        # 5. 加权合成最终M(t)
        m_t = (
            m1_norm * p['dynamic_m_csi300_w_trend'] +
            m2_norm * p['dynamic_m_csi300_w_momentum'] +
            m3_norm * p['dynamic_m_csi300_w_volatility'] +
            m4_norm * p['dynamic_m_csi300_w_turnover']
        )
        
        # 6. 缓存最终的 M(t) 值
        DailyFactorValues.objects.update_or_create(
            stock_code_id=MARKET_INDICATOR_CODE,
            trade_date=self.trade_date,
            factor_code_id='dynamic_M_VALUE',
            defaults={'raw_value': Decimal(str(m_t)), 'norm_score': Decimal(str(m_t))}
        )
        logger.info(f"M(t) 计算完成，值为: {m_t:.4f}，并已存入缓存。")
        return m_t


    def _norm_absolute(self, value, p10, p50, p90, direction):
        """使用固定分位锚点进行绝对标准化（分段线性插值）"""
        value = float(value)
        p10, p50, p90 = float(p10), float(p50), float(p90)
        
        if direction == 'negative':
            value = -value
            p10, p50, p90 = -p90, -p50, -p10
        
        if value <= p10: return -1.0
        if value >= p90: return 1.0
        
        if value <= p50:
            # 在 [-1, 0] 区间插值
            if (p50 - p10) < 1e-9: return 0.0
            return -1 + (value - p10) / (p50 - p10)
        else: # value > p50
            # 在 [0, 1] 区间插值
            if (p90 - p50) < 1e-9: return 0.0
            return (value - p50) / (p90 - p50)

    def _save_m_raw_values_to_cache(self, df: pd.DataFrame):
        """将新计算的M指标原始值批量存入数据库"""
        records = []
        for trade_date, row in df.iterrows():
            for factor_code, raw_value in row.items():
                if pd.notna(raw_value):
                    records.append(DailyFactorValues(
                        stock_code_id=MARKET_INDICATOR_CODE,
                        trade_date=trade_date,
                        factor_code_id=factor_code,
                        raw_value=Decimal(str(raw_value)),
                        norm_score=Decimal(str(raw_value)) # 原始值缓存，norm_score字段可复用
                    ))
        if records:
            DailyFactorValues.objects.bulk_create(records, ignore_conflicts=True)
            logger.debug(f"成功将 {len(records)} 条M(t)基础指标原始值存入缓存。")

    def _calculate_dynamic_weights(self, M_t: float) -> dict:
        """根据M(t)计算四个策略维度的动态权重"""
        logger.debug(f"根据 M(t)={M_t:.4f} 计算动态权重...")
        p = self.dynamic_params
        
        # a. 计算各维度吸引力 A_i
        # A_MT = p['dynamic_c_MT'] * M_t
        # A_BO = p['dynamic_c_BO'] * M_t
        # A_QD = p['dynamic_c_QD'] * (-M_t)
        # A_MR = p['dynamic_c_MR'] * np.exp(- (M_t / p['dynamic_sigma_MR'])**2)
        A_MT = p['dynamic_c_MT'] * (-M_t) 
        A_BO = p['dynamic_c_BO'] * (-M_t)
        # 当M(t)为+1(狂热)时，QD吸引力最大；当M(t)为-1(恐慌)时，吸引力最小
        A_QD = p['dynamic_c_QD'] * M_t
        # MR的逻辑保持不变，它在M(t)接近0（中性/转折期）时吸引力最大，这仍然是合理的
        A_MR = p['dynamic_c_MR'] * np.exp(- (M_t / p['dynamic_sigma_MR'])**2)
        
        # b. 通过Softmax计算最终权重 N_i
        tau = p['dynamic_tau']
        exp_A_MT = np.exp(A_MT / tau)
        exp_A_BO = np.exp(A_BO / tau)
        exp_A_QD = np.exp(A_QD / tau)
        exp_A_MR = np.exp(A_MR / tau)
        
        sum_exp_A = exp_A_MT + exp_A_BO + exp_A_QD + exp_A_MR
        
        weights = {
            'MT': exp_A_MT / sum_exp_A,
            'BO': exp_A_BO / sum_exp_A,
            'QD': exp_A_QD / sum_exp_A,
            'MR': exp_A_MR / sum_exp_A,
        }
        logger.debug(f"动态权重计算完成: MT={weights['MT']:.2%}, BO={weights['BO']:.2%}, MR={weights['MR']:.2%}, QD={weights['QD']:.2%}")
        return weights

    def _calculate_dimension_scores(self, norm_scores_df: pd.DataFrame) -> pd.DataFrame:
        """计算四个策略维度的得分"""
        logger.debug("计算各策略维度得分...")
        p = self.dynamic_params
        scores = pd.DataFrame(index=norm_scores_df.index)

        scores['Score_MT'] = (
            norm_scores_df['dynamic_MA20_SLOPE'] * p['dynamic_k_MT1'] +
            norm_scores_df['dynamic_MA_SCORE'] * p['dynamic_k_MT2'] +
            norm_scores_df['dynamic_ADX_CONFIRM'] * p['dynamic_k_MT3']
        )
        scores['Score_BO'] = (
            norm_scores_df['dynamic_BREAKOUT_PWR'] * p['dynamic_k_BO1'] +
            norm_scores_df['dynamic_VOLUME_SURGE'] * p['dynamic_k_BO2'] +
            norm_scores_df['dynamic_MOM_ACCEL'] * p['dynamic_k_BO3']
        )
        scores['Score_MR'] = (
            norm_scores_df['dynamic_RSI_OS'] * p['dynamic_k_MR1'] +
            norm_scores_df['dynamic_NEG_DEV'] * p['dynamic_k_MR2'] +
            norm_scores_df['dynamic_BOLL_LB'] * p['dynamic_k_MR3']
        )
        scores['Score_QD'] = (
            norm_scores_df['dynamic_LOW_VOL'] * p['dynamic_k_QD1'] +
            norm_scores_df['dynamic_MAX_DD'] * p['dynamic_k_QD2'] +
            norm_scores_df['dynamic_DOWNSIDE_RISK'] * p['dynamic_k_QD3']
        )
        return scores

    def _calculate_final_dynamic_score(self, dimension_scores_df: pd.DataFrame, weights: dict) -> pd.Series:
        """计算最终的 f_dynamic 得分"""
        logger.debug("计算最终综合得分 f_dynamic(x, t)...")
        final_score = (
            dimension_scores_df['Score_MT'] * weights['MT'] +
            dimension_scores_df['Score_BO'] * weights['BO'] +
            dimension_scores_df['Score_MR'] * weights['MR'] +
            dimension_scores_df['Score_QD'] * weights['QD']
        )
        return final_score.sort_values(ascending=False)

    # endregion

    # region: --- 3. 因子计算 (模块化) ---

    def _calculate_all_dynamic_factors(self) -> pd.DataFrame:
        """调度所有12个因子计算方法"""
        logger.debug("开始计算所有动态因子...")
        
        factor_calculators = {
            'dynamic_MA20_SLOPE': self._calc_factor_ma20_slope,
            'dynamic_MA_SCORE': self._calc_factor_ma_score,
            'dynamic_ADX_CONFIRM': self._calc_factor_adx_confirm,
            'dynamic_BREAKOUT_PWR': self._calc_factor_breakout_pwr,
            'dynamic_VOLUME_SURGE': self._calc_factor_volume_surge,
            'dynamic_MOM_ACCEL': self._calc_factor_mom_accel,
            'dynamic_RSI_OS': self._calc_factor_rsi_os,
            'dynamic_NEG_DEV': self._calc_factor_neg_dev,
            'dynamic_BOLL_LB': self._calc_factor_boll_lb,
            'dynamic_LOW_VOL': self._calc_factor_low_vol,
            'dynamic_MAX_DD': self._calc_factor_max_dd,
            'dynamic_DOWNSIDE_RISK': self._calc_factor_downside_risk,
        }
        
        all_factors = {}
        for code, func in factor_calculators.items():
            if code in self.dynamic_factor_defs:
                logger.debug(f"  - 计算因子: {code}")
                all_factors[code] = func()
        
        raw_factors_df = pd.DataFrame(all_factors)
        original_count = len(raw_factors_df)

        inf_count = np.isinf(raw_factors_df).sum().sum()
        if inf_count > 0:
            logger.debug(f"发现 {inf_count} 个无穷大(inf)值，将替换为NaN进行剔除。")
            raw_factors_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # --- 核心调试代码 ---
        if original_count > 0:
            logger.debug("--- NaN 值分析开始 ---")
            nan_counts = raw_factors_df.isna().sum()
            logger.debug(f"因子计算后，各因子的 NaN 值数量 (总计 {original_count} 只股票):")
            logger.debug("\n" + nan_counts.to_string())
            
            # 找出NaN数量最多的因子
            problematic_factors = nan_counts[nan_counts > 0].sort_values(ascending=False)
            if not problematic_factors.empty:
                logger.debug(f"问题可能出在以下因子中，它们的NaN数量较多: \n{problematic_factors}")
            else:
                logger.debug("太棒了！没有任何因子产生NaN。")
            # 检查是否存在某一行全是NaN
            all_nan_rows = raw_factors_df.isna().all(axis=1).sum()
            if all_nan_rows > 0:
                logger.debug(f"警告：有 {all_nan_rows} 只股票的所有因子值都为 NaN！")
            logger.debug("--- NaN 值分析结束 ---")
        raw_factors_df.dropna(inplace=True)
        final_count = len(raw_factors_df)
        logger.debug(f"因子计算完成。因数据不足(NaN)剔除了 {original_count - final_count} 只股票。剩余 {final_count} 只。")
        
        return raw_factors_df

    # --- MT 因子 ---
    def _calc_factor_ma20_slope(self) -> pd.Series:
        """
        计算20日均线的20日线性回归斜率。
        使用numpy.linalg.lstsq进行向量化计算，避免了低效的rolling.apply()。
        """
        ma20 = self.panel_hfq_close.rolling(20).mean()
        
        # 我们只需要最后20个MA20值来计算当前斜率
        last_20_ma20 = ma20.tail(20)
        
        # 如果数据不足20天，则无法计算
        if len(last_20_ma20) < 20:
            # 返回一个全为NaN的Series，保持与原输出结构一致
            return pd.Series(np.nan, index=self.panel_hfq_close.columns)
        # 准备最小二乘法求解
        # x是时间自变量 [0, 1, 2, ..., 19]
        x = np.arange(20)
        # 我们需要一个常数项，所以构建一个 (20, 2) 的矩阵A
        A = np.vstack([x, np.ones(20)]).T
        
        # y是因变量，即每个股票的最后20个MA20值
        y = last_20_ma20.values
        
        # np.linalg.lstsq 会为y的每一列（每个股票）解出 Ax = y 中的 x
        # 返回结果的第一个元素是解的矩阵，其中第一行是斜率(m)，第二行是截距(c)
        slopes, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 将结果转换回pandas Series
        return pd.Series(slopes, index=last_20_ma20.columns)

    def _calc_factor_ma_score(self) -> pd.Series:
        close, ma5, ma10, ma20 = (
            self.panel_hfq_close.iloc[-1],
            self.panel_hfq_close.rolling(5).mean().iloc[-1],
            self.panel_hfq_close.rolling(10).mean().iloc[-1],
            self.panel_hfq_close.rolling(20).mean().iloc[-1]
        )
        return (close > ma5).astype(int) + (ma5 > ma10).astype(int) + (ma10 > ma20).astype(int)

    # def _calc_factor_adx_confirm(self) -> pd.Series:
    #     adx_df = self.panel_hfq_close.apply(lambda s: ta.adx(self.panel_high[s.name], self.panel_low[s.name], s, length=14).iloc[-1])
    #     adx_df = adx_df.T
    #     condition = (adx_df['ADX_14'] > 20) & (adx_df['DMP_14'] > adx_df['DMN_14'])
    #     return adx_df['ADX_14'].where(condition, 0)

    def _calc_factor_adx_confirm(self) -> pd.Series:
        # --- 准备基础数据 ---
        high = self.panel_high
        low = self.panel_low
        close = self.panel_hfq_close
        length = 14
        
        # --- 1. 向量化计算 +DM 和 -DM ---
        # 使用 .diff() 一次性计算所有股票的 up_move 和 down_move
        move_up = high.diff()
        move_down = -low.diff()
        
        # 使用布尔掩码和 .where() 实现条件逻辑
        plus_dm = move_up.where((move_up > move_down) & (move_up > 0), 0.0)
        minus_dm = move_down.where((move_down > move_up) & (move_down > 0), 0.0)
        # --- 2. 向量化计算真实波幅 (TR) ---
        prev_close = close.shift(1)
        range1 = high - low
        range2 = (high - prev_close).abs()
        range3 = (low - prev_close).abs()
        true_range = np.maximum(np.maximum(range1, range2), range3)
        
        # --- 3. 向量化平滑处理 ---
        # 使用 .ewm() 一次性计算所有股票的平滑值 (Wilder's Smoothing)
        alpha = 1 / length
        min_p = length
        
        atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=min_p).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=min_p).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=min_p).mean()
        
        # --- 4. 向量化计算 +DI 和 -DI ---
        # 加上 epsilon 防止除以零
        epsilon = 1e-9
        plus_di = 100 * (plus_dm_smooth / (atr + epsilon))
        minus_di = 100 * (minus_dm_smooth / (atr + epsilon))
        
        # --- 5. 向量化计算 DX ---
        di_sum = plus_di + minus_di
        di_diff_abs = (plus_di - minus_di).abs()
        dx = 100 * (di_diff_abs / (di_sum + epsilon))
        
        # --- 6. 向量化计算最终 ADX ---
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=min_p).mean()
        
        # --- 7. 提取最后一日的数据并应用过滤条件 ---
        # 直接从计算出的面板中提取最后一行，得到的就是所有股票的最新指标
        adx_final = adx.iloc[-1]
        plus_di_final = plus_di.iloc[-1]
        minus_di_final = minus_di.iloc[-1]
        
        # 条件判断现在是在Series上进行，完全向量化
        condition = (adx_final > 20) & (plus_di_final > minus_di_final)
        
        # .where() 也是一个高效的向量化操作
        # 返回的Series结构与原函数完全一致
        return adx_final.where(condition, 0.0)
        

    # --- BO 因子 ---
    # def _calc_factor_breakout_pwr(self) -> pd.Series:
    #     close = self.panel_hfq_close
    #     atr14 = self.panel_hfq_close.apply(lambda s: ta.atr(self.panel_high[s.name], self.panel_low[s.name], s, length=14).iloc[-1])
    #     breakout_level = close.iloc[-60:-1].max()
    #     return (close.iloc[-1] - breakout_level) / (atr14+ 1e-9)

    def _calc_factor_breakout_pwr(self) -> pd.Series:
        # 准备基础数据，与之前相同
        # 1. 准备所有需要的基础数据面板
        hfq_close = self.panel_hfq_close
        raw_close = self.panel_close  # 新增依赖：不复权收盘价
        raw_high = self.panel_high
        raw_low = self.panel_low
        
        # --- 核心修正：将 high 和 low 价格调整到后复权空间 ---
        # 2. 计算每日的复权因子。这是一个全DataFrame的向量化操作。
        #    为防止除以0（尽管在股价中罕见），增加一个极小值。
        adjustment_factor = hfq_close / (raw_close + 1e-9)
        
        # 3. 应用复权因子，得到后复权的 high 和 low。这同样是向量化操作。
        hfq_high = raw_high * adjustment_factor
        hfq_low = raw_low * adjustment_factor
        # --- 修正结束 ---
        # --- ATR 向量化计算开始 (现在使用完全一致的后复权价格) ---
        # 4. 计算前一日的后复权收盘价
        #    注意：这里我们用 hfq_close，因为ATR公式需要前一天的收盘价
        prev_hfq_close = hfq_close.shift(1)
        
        # 5. 计算TR的三个组成部分，现在所有价格都在同一复权空间内
        range1 = hfq_high - hfq_low
        range2 = (hfq_high - prev_hfq_close).abs()
        range3 = (hfq_low - prev_hfq_close).abs()
        
        # 6. 计算真实波幅 (True Range)，完全向量化
        true_range = np.maximum(range1, range2)
        true_range = np.maximum(true_range, range3)
        
        # 7. 计算 ATR，一次性对整个DataFrame进行
        atr_panel = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # 8. 获取最后一天的ATR值
        atr14 = atr_panel.iloc[-1]
        # --- ATR 向量化计算结束 ---
        # 9. 后续计算与之前完全相同，但现在 ATR 的值是基于正确逻辑得出的
        breakout_level = hfq_close.iloc[-60:-1].max()
        return (hfq_close.iloc[-1] - breakout_level) / (atr14 + 1e-9)
        

    def _calc_factor_volume_surge(self) -> pd.Series:
        vol = self.panel_volume
        return vol.iloc[-1] / (vol.iloc[-20:-1].mean()+ 1e-9)

    def _calc_factor_mom_accel(self) -> pd.Series:
        """计算动能加速度：(今日ROC5 / 11日前ROC5) - 1"""
        # 1. 计算整个面板的5日变化率 (动能)
        roc5 = self.panel_hfq_close.pct_change(5, fill_method=None)
        # 2. 使用 shift() 获取11个周期前的动能值。这才是正确的向量化操作。
        # shift() 会在每个股票代码（列）内部独立进行数据下移。
        roc5_shifted = roc5.shift(11)
        # 3. 计算动能加速度
        # 这个计算现在是正确的元素对元素操作：每个格子的值除以它上面11个格子的值
        acceleration = roc5 / (roc5_shifted + 1e-9) - 1
        # 4. 如果数据长度不足，acceleration的开头几行会是NaN，但最后一行应该是有效的。
        # 我们返回最后一行的Series即可。
        if acceleration.empty:
            return pd.Series(np.nan, index=self.panel_hfq_close.columns)
        
        return acceleration.iloc[-1]

    # --- MR 因子 ---
    # def _calc_factor_rsi_os(self) -> pd.Series:
    #     return self.panel_hfq_close.apply(lambda s: ta.rsi(s, length=14).iloc[-1])

    def _calc_factor_rsi_os(self) -> pd.Series:
        close = self.panel_hfq_close
        length = 14 # RSI周期
        
        # --- RSI 向量化计算开始 ---
        # 1. 计算所有股票的价格日度变动。这是一个全DataFrame操作。
        delta = close.diff(1)
        
        # 2. 从delta中分离出上涨(gain)和下跌(loss)的DataFrame。
        # 上涨部分：将负值和0裁剪为0。
        gain = delta.clip(lower=0)
        # 下跌部分：将正值和0裁剪为0，然后取绝对值。
        loss = delta.clip(upper=0).abs()
        
        # 3. 计算平均上涨和平均下跌。使用ewm进行指数平滑，这与Wilder's Smoothing等价。
        # `adjust=False` 确保使用递归平滑公式，与技术分析标准一致。
        # `min_periods=length` 确保在有足够数据时才开始计算。
        avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        
        # 4. 计算相对强度 (RS)。为分母增加一个极小值以防止除以零。
        rs = avg_gain / (avg_loss + 1e-9)
        
        # 5. 根据RS计算RSI。所有操作都是元素级别的，速度极快。
        rsi_panel = 100 - (100 / (1 + rs))
        
        # 6. 获取最后一天的RSI值，得到一个Series，格式与原输出完全一致。
        return rsi_panel.iloc[-1]

    def _calc_factor_neg_dev(self) -> pd.Series:
        ma60 = self.panel_hfq_close.rolling(60).mean().iloc[-1]
        return (self.panel_hfq_close.iloc[-1] - ma60) / (ma60+ 1e-9)

    # def _calc_factor_boll_lb(self) -> pd.Series:
    #     boll_df = self.panel_hfq_close.apply(lambda s: ta.bbands(s, length=20).iloc[-1])
    #     boll_df = boll_df.T
    #     return (self.panel_hfq_close.iloc[-1] - boll_df['BBL_20_2.0']) / (boll_df['BBU_20_2.0'] - boll_df['BBL_20_2.0']+ 1e-9)
    def _calc_factor_boll_lb(self) -> pd.Series:
        # 准备基础数据和参数
        close_panel = self.panel_hfq_close
        length = 20
        std_dev_multiplier = 2.0
        # --- 布林带向量化计算开始 ---
        # 1. 一次性计算所有股票的20日滚动均值 (中轨)
        # .rolling() 返回一个窗口对象，.mean() 在此对象上进行高效计算
        middle_band_panel = close_panel.rolling(window=length, min_periods=length).mean()
        # 2. 一次性计算所有股票的20日滚动标准差
        std_dev_panel = close_panel.rolling(window=length, min_periods=length).std()
        # 3. 通过DataFrame算术一次性计算上下轨
        upper_band_panel = middle_band_panel + std_dev_multiplier * std_dev_panel
        lower_band_panel = middle_band_panel - std_dev_multiplier * std_dev_panel
        # --- 布林带向量化计算结束 ---
        # 4. 从面板数据中直接获取最后一日的截面数据 (Series)
        last_close = close_panel.iloc[-1]
        last_upper_band = upper_band_panel.iloc[-1]
        last_lower_band = lower_band_panel.iloc[-1]
        # 5. 最终因子计算，所有操作都是在Series上进行的向量化运算
        band_width = last_upper_band - last_lower_band
        
        # 加上 epsilon 防止除以零
        return (last_close - last_lower_band) / (band_width + 1e-9)
        

    # --- QD 因子 ---
    def _calc_factor_low_vol(self) -> pd.Series:
        return self.panel_hfq_close.pct_change(fill_method=None).rolling(20, min_periods=2).std().iloc[-1]
    

    def _calc_factor_max_dd(self) -> pd.Series:
        roll_max = self.panel_hfq_close.rolling(60, min_periods=1).max()
        daily_dd = self.panel_hfq_close / roll_max - 1.0
        return daily_dd.rolling(60, min_periods=1).min().iloc[-1]

    def _calc_factor_downside_risk(self) -> pd.Series:
        returns = self.panel_hfq_close.pct_change(fill_method=None)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        # 添加 min_periods=2
        downside_risk = downside_returns.rolling(60, min_periods=2).std() 
        return downside_risk.iloc[-1]

    # endregion
    
    # region: --- 4. 辅助方法 (基本与旧版兼容) ---

    def _log_to_db(self, level, message):
        return
        #SystemLog.objects.create(log_level=level, module_name=MODULE_NAME, message=message)

    def _load_dynamic_parameters_and_defs(self):
        logger.debug("加载动态策略参数和因子定义...")
        params_qs = StrategyParameters.objects.filter(param_name__startswith='dynamic_')
        self.dynamic_params = {p.param_name: float(p.param_value) for p in params_qs}
        
        defs_qs = FactorDefinitions.objects.filter(is_active=True, factor_code__startswith='dynamic_')
        self.dynamic_factor_defs = {f.factor_code: {'direction': f.direction} for f in defs_qs}
        logger.debug(f"加载了 {len(self.dynamic_params)} 个动态参数和 {len(self.dynamic_factor_defs)} 个启用的动态因子定义。")

    def _get_market_trade_dates(self, lookback_period: int) -> list[date]:
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
        logger.debug("开始执行初步筛选...")
        all_stocks = StockInfo.objects.filter(status=StockInfo.StatusChoices.LISTING)
        #过滤科创版、ST、*ST
        non_st_stocks = all_stocks.exclude(stock_code__contains='.688').exclude(stock_name__startswith='ST').exclude(stock_name__startswith='*ST')
        
        min_listing_date = self.trade_date - timedelta(days=int(self.dynamic_params['dynamic_lookback_new_stock']))
        
        non_new_stocks = non_st_stocks.filter(listing_date__lt=min_listing_date)
        
        stock_pool_codes = list(non_new_stocks.values_list('stock_code', flat=True))
        logger.debug(f"剔除科创版和ST和次新股后，剩余 {len(stock_pool_codes)} 只股票。")

        lookback_days = 20
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
        recent_trade_dates = sorted(quotes_df['trade_date'].unique())[-lookback_days:]
        quotes_df = quotes_df[quotes_df['trade_date'].isin(recent_trade_dates)]

        avg_turnover = quotes_df.groupby('stock_code_id')['turnover'].mean()
        liquid_stocks = avg_turnover[avg_turnover >= self.dynamic_params['dynamic_min_liquidity']]
        
        final_stock_pool = list(liquid_stocks.index)
        final_stock_pool=stock_pool_codes
        logger.debug(f"剔除低流动性股后，最终剩余 {len(final_stock_pool)} 只股票进入精选池。")
        
        return final_stock_pool

    def _load_market_data(self, stock_pool: list[str], for_m_calc: bool = False):
        max_lookback=1000
        if for_m_calc:
            # M(t)本身要看750天历史，其计算又需要60天窗口，所以总共需要约810天
            # 从数据库动态获取参数，避免硬编码
            m_lookback_param = int(self.dynamic_params.get('dynamic_m_lookback', 750))
            # 额外增加90天作为计算Buffer (覆盖60个交易日)
            max_lookback = m_lookback_param + 90 
        else:
            # 个股因子最长回溯60天，给足buffer到250天是合理的
            max_lookback = 250
        logger.debug(f"确定最大数据回溯期为 {max_lookback} 个交易日。")

        trade_dates = self._get_market_trade_dates(max_lookback)
        if not trade_dates: raise ValueError("无法获取市场交易日历。")
        
        logger.debug(f"正在加载 {len(stock_pool)} 只股票在 {len(trade_dates)} 个交易日内的行情数据...")
        
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool,
            trade_date__in=trade_dates
        ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close')
        
        if not quotes_qs: raise ValueError("在指定日期范围内未找到任何股票的行情数据。")

        df = pd.DataFrame.from_records(quotes_qs)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        logger.debug("正在构建面板数据(Panel Data)...")
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']:
            panel = df.pivot(index='trade_date', columns='stock_code_id', values=col).astype(float)
            setattr(self, f'panel_{col}', panel)
        logger.debug("面板数据构建完成。")

    def _standardize_factors(self, raw_factors_df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("开始对因子值进行标准化...")
        norm_scores_df = pd.DataFrame(index=raw_factors_df.index)
        
        for factor_code, series in raw_factors_df.items():
            direction = self.dynamic_factor_defs[factor_code]['direction']
            p1, p99 = series.quantile(0.01), series.quantile(0.99)
            
            if (p99 - p1) < 1e-9:
                norm_scores_df[factor_code] = 0
                continue
            
            x_prime = series.clip(p1, p99)
            
            if direction == 'positive':
                score = ((x_prime - p1) / (p99 - p1)) * 200 - 100
            else:
                score = ((p99 - x_prime) / (p99 - p1)) * 200 - 100
            
            norm_scores_df[factor_code] = score
            
        logger.debug("因子标准化完成。")
        return norm_scores_df

    def _generate_trading_plan(self, final_scores: pd.Series, dimension_scores: pd.DataFrame) -> pd.DataFrame:
        """
        [V2.1 - Patched] 为Top N股票生成动态的、基于策略DNA的交易预案。
        - 修复了ATR计算中混合使用复权与不复权价格的严重BUG。
        - 优化了代码可读性。
        """
        logger.debug("开始生成动态交易预案 (V2.1 - Patched)...")
        top_n = int(self.dynamic_params.get('dynamic_top_n', 10))
        top_stocks_scores = final_scores.head(top_n)
        
        if top_stocks_scores.empty:
            return pd.DataFrame()
 
        top_stock_codes = top_stocks_scores.index.tolist()
        
        # 提取Top N股票的维度得分
        top_dimension_scores = dimension_scores.reindex(top_stock_codes)
 
        # === 核心修正点在这里 ===
        # 使用不复权的 self.panel_close 作为 apply 的主体，确保 lambda 中的 's' 是不复权收盘价序列
        logger.debug("正在基于不复权价格计算ATR...")
        atr14 = self.panel_close.apply(
            lambda s: ta.atr(
                high=self.panel_high[s.name].astype(float), 
                low=self.panel_low[s.name].astype(float), 
                close=s.astype(float),  # 这里的 's' 现在是正确的不复权收盘价序列
                length=14
            ).iloc[-1]
        )
        # === 修正结束 ===
 
        # 准备后续计算所需的数据
        last_close_series = self.panel_close.iloc[-1].reindex(top_stock_codes)
        last_atr_series = atr14.reindex(top_stock_codes)
        p = self.dynamic_params
        
        plans = []
        for stock_code in top_stock_codes:
            # --- 核心动态逻辑开始 (这部分逻辑是正确的，无需修改) ---
            
            # 步骤1: 计算各维度的贡献值 (DCV)
            # ... (此部分逻辑保持不变)
            dcv = {
                'MT': self.dynamic_weights['MT'] * top_dimension_scores.loc[stock_code, 'Score_MT'],
                'BO': self.dynamic_weights['BO'] * top_dimension_scores.loc[stock_code, 'Score_BO'],
                'MR': self.dynamic_weights['MR'] * top_dimension_scores.loc[stock_code, 'Score_MR'],
                'QD': self.dynamic_weights['QD'] * top_dimension_scores.loc[stock_code, 'Score_QD'],
            }
            
            # 步骤2: 提取正向贡献值 (PDCV) 并计算策略DNA权重 (SSW)
            # ... (此部分逻辑保持不变)
            pdcv = {k: max(0, v) for k, v in dcv.items()}
            total_pdcv = sum(pdcv.values())
            
            if total_pdcv <= 1e-9:
                logger.warning(f"股票 {stock_code} 信号混乱 (Total_PDCV <= 0)，采用保守的QD策略作为其开盘区间。")
                ssw = {'MT': 0, 'BO': 0, 'MR': 0, 'QD': 1.0}
            else:
                ssw = {k: v / total_pdcv for k, v in pdcv.items()}
            strategy_dna_str = "|".join([f"{k}:{v:.2f}" for k, v in ssw.items()])
            # 步骤3: 加权合成动态k值
            # ... (此部分逻辑保持不变)
            k_gap_dynamic = (
                ssw['MT'] * p['dynamic_miopmaop_k_gap_base_mt'] +
                ssw['BO'] * p['dynamic_miopmaop_k_gap_base_bo'] +
                ssw['MR'] * p['dynamic_miopmaop_k_gap_base_mr'] +
                ssw['QD'] * p['dynamic_miopmaop_k_gap_base_qd']
            )
            k_drop_dynamic = (
                ssw['MT'] * p['dynamic_miopmaop_k_drop_base_mt'] +
                ssw['BO'] * p['dynamic_miopmaop_k_drop_base_bo'] +
                ssw['MR'] * p['dynamic_miopmaop_k_drop_base_mr'] +
                ssw['QD'] * p['dynamic_miopmaop_k_drop_base_qd']
            )
            # --- 核心动态逻辑结束 ---
 
            # 获取该股票的不复权收盘价和正确的ATR值
            close = last_close_series.get(stock_code)
            atr = last_atr_series.get(stock_code)
 
            if pd.isna(close) or pd.isna(atr) or close <= 0:
                logger.warning(f"股票 {stock_code} 的收盘价或ATR无效(可能数据不足)，无法生成预案。")
                continue
 
            # 使用简化的公式，更清晰且能避免浮点精度问题
            miop = close - k_drop_dynamic * atr
            maop = close + k_gap_dynamic * atr
 
            plans.append({
                'stock_code': stock_code,
                'rank': len(plans) + 1,
                'final_score': top_stocks_scores.get(stock_code),
                'miop': miop,
                'maop': maop,
                'strategy_dna': strategy_dna_str
            })
            
            # 更新日志，现在ATR值会是正常的
            logger.debug(f"预案: {stock_code}, Rank: {len(plans)}, ATR:{atr:.2f}, "
                        f"SSW(MT/BO/MR/QD):({ssw['MT']:.2f}/{ssw['BO']:.2f}/{ssw['MR']:.2f}/{ssw['QD']:.2f}), "
                        f"k_gap_dyn:{k_gap_dynamic:.2f}, k_drop_dyn:{k_drop_dynamic:.2f}, MIOP:{miop:.2f}, MAOP:{maop:.2f}")
 
        if not plans:
            return pd.DataFrame()
 
        plan_df = pd.DataFrame(plans).dropna(subset=['miop', 'maop'])
        logger.debug("动态交易预案生成完成。")
        return plan_df

    @transaction.atomic
    def _save_results(self, raw_factors_df, norm_scores_df, trading_plan_df):
        logger.debug("开始将结果保存到数据库...")
        
        # 1. 保存每日因子值
        factor_values_to_create = []
        for stock_code, row in raw_factors_df.iterrows():
            for factor_code, raw_value in row.items():
                norm_score = norm_scores_df.loc[stock_code, factor_code]
                factor_values_to_create.append(
                    DailyFactorValues(
                        stock_code_id=stock_code, trade_date=self.trade_date,
                        factor_code_id=factor_code, raw_value=Decimal(str(raw_value)),
                        norm_score=Decimal(str(norm_score))
                    )
                )
        DailyFactorValues.objects.bulk_create(factor_values_to_create, ignore_conflicts=True)

        # 2. 保存每日交易预案
        plan_date = self.trade_date + timedelta(days=1)
        DailyTradingPlan.objects.filter(plan_date=plan_date).delete()
        
        plans_to_create = []
        for _, row in trading_plan_df.iterrows():
            plans_to_create.append(
                DailyTradingPlan(
                    plan_date=plan_date, stock_code_id=row['stock_code'],
                    rank=row['rank'], final_score=Decimal(str(row['final_score'])),
                    miop=Decimal(str(row['miop'])).quantize(Decimal('0.01')),
                    maop=Decimal(str(row['maop'])).quantize(Decimal('0.01')),
                    status=DailyTradingPlan.StatusChoices.PENDING,
                    strategy_dna=row['strategy_dna']
                )
            )
        DailyTradingPlan.objects.bulk_create(plans_to_create)
        
        log_message = f"T-1日({self.trade_date})动态选股完成, T日({plan_date})预案如下:\n"
        log_message += trading_plan_df.to_string(index=False)
        self._log_to_db('INFO', log_message)
        logger.debug("所有结果已成功保存到数据库。")

    # endregion
