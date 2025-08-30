# selection_manager/service/selection_service.py

import logging
from datetime import date, timedelta
from decimal import Decimal
from django.db.models import F # 用于类型转换
from common.models import IndexQuotesCsi300
from .m_value_service import m_value_service_instance

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

    def __init__(self, trade_date: date, mode: str = 'realtime' , one_strategy: str = None, preloaded_panels: dict = None):
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
        self.one_strategy = one_strategy
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

        # --- 性能优化：处理预加载数据 ---
        if preloaded_panels:
            logger.debug("检测到预加载面板数据，直接赋值。")
            self.panel_open = preloaded_panels.get('open')
            self.panel_high = preloaded_panels.get('high')
            self.panel_low = preloaded_panels.get('low')
            self.panel_close = preloaded_panels.get('close')
            self.panel_volume = preloaded_panels.get('volume')
            self.panel_turnover = preloaded_panels.get('turnover')
            self.panel_hfq_close = preloaded_panels.get('hfq_close')
        # --- 结束 ---

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
            {'factor_code': 'dynamic_ADX_CONFIRM', 'factor_name': '动态-ADX趋势确认', 'direction': 'positive'},
            {'factor_code': 'dynamic_v2_MA_SLOPE', 'factor_name': '动态V2-均线斜率因子', 'direction': 'positive'},
            {'factor_code': 'dynamic_v2_MA_SCORE', 'factor_name': '动态V2-均线排列评分', 'direction': 'positive'},
            {'factor_code': 'dynamic_v2_CPC_Factor', 'factor_name': '动态V2-K线路径一致性', 'direction': 'positive'},
            {'factor_code': 'dynamic_v2_VPCF', 'factor_name': '动态V2-量价协同因子', 'direction': 'positive'},
            
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
            # 'dynamic_m_csi300_anchor_trend_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '趋势强度-P10锚点'},
            # 'dynamic_m_csi300_anchor_trend_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '趋势强度-P50锚点'},
            # 'dynamic_m_csi300_anchor_trend_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '趋势强度-P90锚点'},
            # 'dynamic_m_csi300_anchor_momentum_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '短期动能-P10锚点'},
            # 'dynamic_m_csi300_anchor_momentum_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '短期动能-P50锚点'},
            # 'dynamic_m_csi300_anchor_momentum_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '短期动能-P90锚点'},
            # 'dynamic_m_csi300_anchor_volatility_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '波动水平-P10锚点'},
            # 'dynamic_m_csi300_anchor_volatility_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '波动水平-P50锚点'},
            # 'dynamic_m_csi300_anchor_volatility_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '波动水平-P90锚点'},
            # 'dynamic_m_csi300_anchor_turnover_p10': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '量能状态-P10锚点'},
            # 'dynamic_m_csi300_anchor_turnover_p50': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '量能状态-P50锚点'},
            # 'dynamic_m_csi300_anchor_turnover_p90': {'value': '0', 'group': 'M_CSI300_ANCHORS', 'desc': '量能状态-P90锚点'},

            # 动态权重 N_i(M(t)) 参数
            'dynamic_c_MT': {'value': '1.3', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 趋势动能'},
            'dynamic_c_BO': {'value': '0.7', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 强势突破'},
            'dynamic_c_QD': {'value': '1', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 质量防御'},
            'dynamic_c_MR': {'value': '1.5', 'group': 'N_ATTRACTION', 'desc': '吸引力函数强度系数: 均值回归'},
            'dynamic_sigma_MR': {'value': '0.25', 'group': 'N_PARAMS', 'desc': '均值回归策略适用范围宽度'},
            'dynamic_tau': {'value': '0.5', 'group': 'N_PARAMS', 'desc': 'Softmax温度系数(控制切换灵敏度)'},

            # 维度内部因子权重 (k_ij)
            'dynamic_k_MT1': {'value': '0.5', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-MA20斜率权重'},
            'dynamic_k_MT2': {'value': '0.3', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-均线排列权重'},
            'dynamic_k_MT3': {'value': '0.2', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-ADX确认权重'},
            'dynamic_k_MT_ADX_CONFIRM': {'value': '0.3', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能-ADX确认权重'},
            'dynamic_k_MT_MA_SLOPE': {'value': '0.1', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能V2-均线斜率权重'},
            'dynamic_k_MT_MA_SCORE': {'value': '0.1', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能V2-均线排列评分权重'},
            'dynamic_k_MT_CPC_Factor': {'value': '0.25', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能V2-K线路径一致性权重'},
            'dynamic_k_MT_VPCF': {'value': '0.25', 'group': 'K_WEIGHTS_MT', 'desc': '趋势动能V2-量价协同因子权重'},
            'dynamic_k_BO1': {'value': '0.45', 'group': 'K_WEIGHTS_BO', 'desc': '强势突破-突破强度权重'},
            'dynamic_k_BO2': {'value': '0.35', 'group': 'K_WEIGHTS_BO', 'desc': '强势突破-成交量激增权重'},
            'dynamic_k_BO3': {'value': '0.2', 'group': 'K_WEIGHTS_BO', 'desc': '强势突破-动能加速度权重'},
            'dynamic_k_MR1': {'value': '0.3', 'group': 'K_WEIGHTS_MR', 'desc': '均值回归-RSI超卖权重'},
            'dynamic_k_MR2': {'value': '0.5', 'group': 'K_WEIGHTS_MR', 'desc': '均值回归-负向偏离度权重'},
            'dynamic_k_MR3': {'value': '0.2', 'group': 'K_WEIGHTS_MR', 'desc': '均值回归-布林下轨支撑权重'},
            'dynamic_k_QD1': {'value': '0.4', 'group': 'K_WEIGHTS_QD', 'desc': '质量防御-低波动率权重'},
            'dynamic_k_QD2': {'value': '0.3', 'group': 'K_WEIGHTS_QD', 'desc': '质量防御-最大回撤权重'},
            'dynamic_k_QD3': {'value': '0.3', 'group': 'K_WEIGHTS_QD', 'desc': '质量防御-下行风险权重'},
            
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
            'dynamic_v2_cpc_n': {'value': '10', 'group': 'FACTOR_PARAMS', 'desc': 'CPC因子-平滑周期N'},
            'dynamic_v2_vpcf_s': {'value': '5', 'group': 'FACTOR_PARAMS', 'desc': 'VPCF因子-短期均线周期s'},
            'dynamic_v2_vpcf_l': {'value': '20', 'group': 'FACTOR_PARAMS', 'desc': 'VPCF因子-长期均线周期l'},
            'dynamic_v2_vpcf_n_smooth': {'value': '5', 'group': 'FACTOR_PARAMS', 'desc': 'VPCF因子-最终平滑周期N_smooth'},
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
            self.market_regime_S = self._calculate_market_regime_S()
            self.dynamic_weights = self._calculate_dynamic_weights(self.market_regime_M,self.market_regime_S)
            if self.panel_close is None:
                self._load_market_data(initial_stock_pool)
            else:
                logger.debug("使用预加载的面板数据，跳过 _load_market_data。")
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

    def _calculate_market_regime_M(self, stock_pool: list[str]) -> float:
        """
        计算市场状态函数 M(t)
        """
        
        # =======================================================================
        # [ML预测接口] - 未来切换到机器学习模型预测M值的入口
        # =======================================================================
        try:
            # 1. 获取最近60个交易日的数据
            quotes_60_days_qs = IndexQuotesCsi300.objects.filter(
                trade_date__lte=self.trade_date
            ).order_by('-trade_date')[:60]
            
            if len(quotes_60_days_qs) < 60:
                logger.warning("沪深300数据不足60天，无法使用ML模型进行预测，将回退到传统方法。")
            else:
                # [修复] 从QuerySet直接构建DataFrame
                df_60_days_raw = pd.DataFrame.from_records(quotes_60_days_qs.values())
                
                # 反转顺序使日期从旧到新
                df_60_days = df_60_days_raw.iloc[::-1].reset_index(drop=True)
                
                # 2. 调用预测服务 (m_value_service内部会处理类型转换)
                ml_m_value = m_value_service_instance.predict_csi300_next_day_trend(df_60_days)
                
                # 3. 【重要】将ML预测结果存入缓存
                DailyFactorValues.objects.update_or_create(
                    stock_code_id=MARKET_INDICATOR_CODE,
                    trade_date=self.trade_date,
                    factor_code_id='dynamic_M_VALUE',
                    defaults={'raw_value': Decimal(str(ml_m_value)), 'norm_score': Decimal(str(ml_m_value))}
                )
                logger.info(f"已使用ML模型预测M(t) = {ml_m_value:.4f}")
                return ml_m_value
        except Exception as e:
            logger.error(f"调用ML模型预测M值时发生错误: {e}", exc_info=True)
    def _calculate_market_regime_M_old(self, stock_pool: list[str]) -> float:
        """
        计算市场状态函数 M(t) - V2.0 沪深300基准版 ----老版本暂时废弃
        使用固定分位锚点法进行绝对标准化，并具备容错机制。
        """
        logger.debug("开始计算市场状态 M(t) [V2.0]...")
        # 1. 检查当日缓存
        try:
            # cached_m = DailyFactorValues.objects.get(
            #     stock_code_id=MARKET_INDICATOR_CODE,
            #     trade_date=self.trade_date,
            #     factor_code_id='dynamic_M_VALUE'
            # )
            # m_value = float(cached_m.raw_value)
            # logger.debug(f"成功从缓存中读取当日 M(t) = {m_value:.4f}")
            # return m_value
            pass
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

    def _calculate_market_regime_S(self) -> float:
        """
        [新增] 计算市场状态变化率 S(t)。
        此方法在 M(t) 计算之后调用。
        """
        logger.debug("开始计算市场状态变化率 S(t)...")
        window_size = int(self.dynamic_params.get('dynamic_s_window_size', 10))
        try:
            # 1. 获取最近N个交易日的M(t)值
            m_values_qs = DailyFactorValues.objects.filter(
                stock_code_id=MARKET_INDICATOR_CODE,
                factor_code_id='dynamic_M_VALUE',
                trade_date__lte=self.trade_date
            ).order_by('-trade_date')[:window_size]
            if len(m_values_qs) < window_size:
                logger.warning(f"M(t)历史数据不足 {window_size} 天，无法计算S(t)，返回0。")
                return 0.0
            # 将查询结果转换为pandas Series，并反转顺序使时间从远到近
            m_values = pd.Series([float(m.raw_value) for m in m_values_qs]).iloc[::-1]
            
            # 2. 进行线性回归
            time_series = np.arange(len(m_values))
            slope, _, _, _, _ = linregress(x=time_series, y=m_values.values)
            # 3. 缓存原始斜率
            DailyFactorValues.objects.update_or_create(
                stock_code_id=MARKET_INDICATOR_CODE,
                trade_date=self.trade_date,
                factor_code_id='dynamic_S_RAW_SLOPE',
                defaults={'raw_value': Decimal(str(slope)), 'norm_score': Decimal(str(slope))}
            )
            
            # 4. 进行绝对标准化
            norm_s = self._norm_absolute_slope(slope)
            # 5. 缓存标准化后的S(t)值
            DailyFactorValues.objects.update_or_create(
                stock_code_id=MARKET_INDICATOR_CODE,
                trade_date=self.trade_date,
                factor_code_id='dynamic_S_VALUE',
                defaults={'raw_value': Decimal(str(norm_s)), 'norm_score': Decimal(str(norm_s))}
            )
            logger.info(f"S(t) 计算完成，原始斜率: {slope:.4f}, 标准化值: {norm_s:.4f}")
            return norm_s
        except Exception as e:
            logger.error(f"计算S(t)时发生错误: {e}", exc_info=True)
            return 0.0

    def _norm_absolute_slope(self, raw_slope: float) -> float:
        """
        [新增] S(t)的绝对标准化函数。
        """
        upper_bound = self.dynamic_params.get('dynamic_s_norm_upper_bound', 0.1)
        lower_bound = self.dynamic_params.get('dynamic_s_norm_lower_bound', -0.1)
        if raw_slope >= upper_bound:
            return 1.0
        if raw_slope <= lower_bound:
            return -1.0
        
        value_range = upper_bound - lower_bound
        if value_range < 1e-9:
            return 0.0
    
        # 线性插值到 [-1, 1]
        return -1.0 + 2.0 * (raw_slope - lower_bound) / value_range

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

    def _calculate_dynamic_weights(self, M_t: float, S_t: float) -> dict:
        """根据M(t)计算四个策略维度的动态权重"""
        logger.debug(f"根据 M(t)={M_t:.4f} 计算动态权重...")
        p = self.dynamic_params
        k=M_t
        # a. 计算各维度吸引力 A_i
        A_MT = p['dynamic_c_MT'] * k
        A_BO = p['dynamic_c_BO'] * k
        # A_QD = p['dynamic_c_QD'] * (-M_t)
        A_QD = p['dynamic_c_QD']
        A_MR = p['dynamic_c_MR'] * np.exp(- (k / p['dynamic_sigma_MR'])**2)
        
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
        epsilon = 1e-9
        if abs(M_t) < epsilon:
            weights = {
            'MT': 0.0,
            'BO': 0.0,
            'QD': 0.0,
            'MR': 0.0,
            'OLD':1.0
            }
        else:

            weights = {
            'MT': 0.0,
            'BO': 0.0,
            'QD': 0.0,
            'MR': 0.0,
            'OLD':1.0
        }

        if self.one_strategy and self.one_strategy in weights:
            logger.debug(f"单策略模式已激活，强制使用策略: {self.one_strategy}")
            for key in weights:
                weights[key] = 0.0
            weights[self.one_strategy] = 1.0
        logger.debug(f"动态权重计算完成: MT={weights['MT']:.2%}, BO={weights['BO']:.2%}, MR={weights['MR']:.2%}, QD={weights['QD']:.2%}, OLD={weights['OLD']:.2%}")
        return weights

    def _calculate_dimension_scores(self, norm_scores_df: pd.DataFrame) -> pd.DataFrame:
        """计算四个策略维度的得分"""
        logger.debug("计算各策略维度得分...")
        p = self.dynamic_params
        scores = pd.DataFrame(index=norm_scores_df.index)

        scores['Score_MT'] = (
            norm_scores_df['dynamic_ADX_CONFIRM'] * p['dynamic_k_MT_ADX_CONFIRM'] +
            norm_scores_df['dynamic_v2_MA_SLOPE'] * p['dynamic_k_MT_MA_SLOPE'] +
            norm_scores_df['dynamic_v2_MA_SCORE'] * p['dynamic_k_MT_MA_SCORE'] +
            norm_scores_df['dynamic_v2_CPC_Factor'] * p['dynamic_k_MT_CPC_Factor'] +
            norm_scores_df['dynamic_v2_VPCF'] * p['dynamic_k_MT_VPCF']#good
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
        scores['Score_OLD'] = (
            self.dynamic_params['w_trend']*(norm_scores_df['MA20_SLOPE'] * self.dynamic_params['k1'] +
            norm_scores_df['MA_ALIGNMENT'] * self.dynamic_params['k2'] +
            norm_scores_df['ADX_TREND'] * self.dynamic_params['k3'])+
            self.dynamic_params['w_momentum']*(norm_scores_df['ROC10'] * self.dynamic_params['k4'] +
            norm_scores_df['VOL_BREAKOUT'] * self.dynamic_params['k5'] +
            norm_scores_df['NEW_HIGH_MOMENTUM'] * self.dynamic_params['k6'])+
            self.dynamic_params['w_quality']*(norm_scores_df['VOLATILITY20'] * self.dynamic_params['k7'] +
            norm_scores_df['LIQUIDITY20'] * self.dynamic_params['k8'])
        )
        # scores['Score_MT'] = norm_scores_df['dynamic_ADX_CONFIRM']
        # scores['Score_BO'] = norm_scores_df['dynamic_v2_MA_SLOPE']
        # scores['Score_MR'] = norm_scores_df['dynamic_v2_MA_SCORE']
        # scores['Score_QD'] = norm_scores_df['dynamic_v2_CPC_Factor']
        # scores['Score_OLD'] = norm_scores_df['dynamic_v2_VPCF']
        return scores

    def _calculate_final_dynamic_score(self, dimension_scores_df: pd.DataFrame, weights: dict) -> pd.Series:
        """计算最终的 f_dynamic 得分"""
        logger.debug("计算最终综合得分 f_dynamic(x, t)...")
        final_score = (
            dimension_scores_df['Score_MT'] * weights['MT'] +
            dimension_scores_df['Score_BO'] * weights['BO'] +
            dimension_scores_df['Score_MR'] * weights['MR'] +
            dimension_scores_df['Score_QD'] * weights['QD'] +
            dimension_scores_df['Score_OLD'] * weights['OLD']
        )
        return final_score.sort_values(ascending=False)

    # endregion

    # region: --- 3. 因子计算 (模块化) ---

    def _calculate_all_dynamic_factors(self) -> pd.DataFrame:
        """调度所有12个因子计算方法"""
        logger.debug("开始计算所有动态因子...")
        
        factor_calculators = {
            'dynamic_ADX_CONFIRM': self._calc_factor_adx_confirm,
            'dynamic_v2_MA_SLOPE': self._calc_factor_v2_ma_slope,
            'dynamic_v2_MA_SCORE': self._calc_factor_v2_ma_score,
            'dynamic_v2_CPC_Factor': self._calc_factor_v2_cpc_factor,
            'dynamic_v2_VPCF': self._calc_factor_v2_vpcf,
            'dynamic_BREAKOUT_PWR': self._calc_factor_breakout_pwr,
            'dynamic_VOLUME_SURGE': self._calc_factor_volume_surge,
            'dynamic_MOM_ACCEL': self._calc_factor_mom_accel,
            'dynamic_RSI_OS': self._calc_factor_rsi_os,
            'dynamic_NEG_DEV': self._calc_factor_neg_dev,
            'dynamic_BOLL_LB': self._calc_factor_boll_lb,
            'dynamic_LOW_VOL': self._calc_factor_low_vol,
            'dynamic_MAX_DD': self._calc_factor_max_dd,
            'dynamic_DOWNSIDE_RISK': self._calc_factor_downside_risk,
            #old
            'MA20_SLOPE': self._calc_factor_x1_ma20_slope,
            'MA_ALIGNMENT': self._calc_factor_x2_ma_alignment,
            'ADX_TREND': self._calc_factor_adx_confirm,
            'ROC10': self._calc_factor_x4_roc,
            'VOL_BREAKOUT': self._calc_factor_x5_vol_breakout,
            'NEW_HIGH_MOMENTUM': self._calc_factor_x6_new_high_momentum,
            'VOLATILITY20': self._calc_factor_x7_volatility,
            'LIQUIDITY20': self._calc_factor_x8_liquidity,
            # 'MA20_SLOPE': self._calc_factor_volume_surge,
            # 'MA_ALIGNMENT': self._calc_factor_volume_surge,
            # 'ADX_TREND': self._calc_factor_volume_surge,
            # 'ROC10': self._calc_factor_volume_surge,
            # 'VOL_BREAKOUT': self._calc_factor_volume_surge,
            # 'NEW_HIGH_MOMENTUM': self._calc_factor_volume_surge,
            # 'VOLATILITY20': self._calc_factor_volume_surge,
            # 'LIQUIDITY20': self._calc_factor_volume_surge,
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
    def _calc_factor_v2_ma_slope(self) -> pd.Series:
        """
        计算MA20在过去20个交易日中每一天的日度变化率的指数移动平均。
        """
        epsilon = 1e-9
        try:
            # 1. 计算MA20
            ma20 = self.panel_hfq_close.rolling(window=20, min_periods=20).mean()
            
            # 2. 计算MA20的日度变化率
            daily_roc = ma20.pct_change(1, fill_method=None)
            
            # 3. 计算变化率的20日指数移动平均
            # 使用 .ewm() 计算，alpha = 2 / (span + 1)
            factor_panel = daily_roc.ewm(span=20, adjust=False, min_periods=20).mean()
            
            # 4. 返回最后一天的值
            return factor_panel.iloc[-1]
        except Exception as e:
            logger.error(f"计算 dynamic_v2_MA_SLOPE 时出错: {e}")
            return pd.Series(np.nan, index=self.panel_hfq_close.columns)
    def _calc_factor_v2_ma_score(self) -> pd.Series:
        """
        计算三个归一化的均线离差的加权和。
        """
        epsilon = 1e-9
        try:
            close = self.panel_hfq_close
            ma5 = close.rolling(window=5, min_periods=5).mean()
            ma10 = close.rolling(window=10, min_periods=10).mean()
            ma20 = close.rolling(window=20, min_periods=20).mean()
            
            # 1. 计算三个归一化的均线离差
            spread1 = (close - ma5) / (ma5 + epsilon)
            spread2 = (ma5 - ma10) / (ma10 + epsilon)
            spread3 = (ma10 - ma20) / (ma20 + epsilon)
            
            # 2. 等权重加权求和
            # 权重可以参数化，但根据文档，这里使用等权重
            factor_panel = 0.333 * spread1 + 0.333 * spread2 + 0.333 * spread3
            
            # 3. 返回最后一天的值
            return factor_panel.iloc[-1]
        except Exception as e:
            logger.error(f"计算 dynamic_v2_MA_SCORE 时出错: {e}")
            return pd.Series(np.nan, index=self.panel_hfq_close.columns)
    def _calc_factor_v2_cpc_factor(self) -> pd.Series:
        """
        计算K线路径一致性 (CPC_Factor)。
        """
        epsilon = 1e-9
        try:
            # 1. 统一价格到后复权空间
            adjustment_factor = self.panel_hfq_close / (self.panel_close + epsilon)
            hfq_high = self.panel_high * adjustment_factor
            hfq_low = self.panel_low * adjustment_factor
            hfq_close = self.panel_hfq_close
            # 2. 计算每日收盘强度 (DCP_t)
            price_range = hfq_high - hfq_low
            # 使用 where 避免除以零
            dcp_panel = np.where(
                price_range > epsilon,
                (2 * hfq_close - hfq_high - hfq_low) / (price_range + epsilon),
                0.0
            )
            # 将numpy数组转回DataFrame以使用ewm
            dcp_panel_df = pd.DataFrame(dcp_panel, index=hfq_close.index, columns=hfq_close.columns)
            # 3. 计算最终因子 (EWMA)
            n_period = int(self.dynamic_params.get('dynamic_v2_cpc_n', 10))
            factor_panel = dcp_panel_df.ewm(span=n_period, adjust=False, min_periods=n_period).mean()
            
            return factor_panel.iloc[-1]
        except Exception as e:
            logger.error(f"计算 dynamic_v2_CPC_Factor 时出错: {e}")
            return pd.Series(np.nan, index=self.panel_hfq_close.columns)
    
    def _calc_factor_v2_vpcf(self) -> pd.Series:
        """
        计算量价协同因子 (VPCF)。
        """
        epsilon = 1e-9
        try:
            # 1. 获取参数
            s = int(self.dynamic_params.get('dynamic_v2_vpcf_s', 10))
            l = int(self.dynamic_params.get('dynamic_v2_vpcf_l', 30))
            n_smooth = int(self.dynamic_params.get('dynamic_v2_vpcf_n_smooth', 10))
            # 2. 计算平滑日度价格动量
            ma_close_s = self.panel_hfq_close.rolling(window=s, min_periods=s).mean()
            price_momentum = ma_close_s.pct_change(1, fill_method=None) # (MA_t / MA_{t-1}) - 1
            # 3. 计算相对成交量水平
            ma_vol_s = self.panel_volume.rolling(window=s, min_periods=s).mean()
            ma_vol_l = self.panel_volume.rolling(window=l, min_periods=l).mean()
            volume_level = (ma_vol_s / (ma_vol_l + epsilon)) - 1
            # 4. 计算日度协同分
            daily_score = price_momentum * volume_level
            # 5. 计算最终因子 (EWMA)
            factor_panel = daily_score.ewm(span=n_smooth, adjust=False, min_periods=n_smooth).mean()
            
            return factor_panel.iloc[-1]
        except Exception as e:
            logger.error(f"计算 dynamic_v2_VPCF 时出错: {e}")
            return pd.Series(np.nan, index=self.panel_hfq_close.columns)
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
        condition = (adx_final > 25) & (plus_di_final > minus_di_final)
        
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

    # --- OLD 因子 ---
    def _calc_factor_x1_ma20_slope(self) -> pd.Series:
        ma_period = self.dynamic_params['lookback_ma20']
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
            lambda col: ta.sma(close=col, length=self.dynamic_params['lookback_ma5'])
        )
        ma10 = self.panel_hfq_close.apply(
            lambda col: ta.sma(close=col, length=self.dynamic_params['lookback_ma10'])
        )
        ma20 = self.panel_hfq_close.apply(
            lambda col: ta.sma(close=col, length=self.dynamic_params['lookback_ma20'])
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
        
        adx_period = self.dynamic_params['lookback_adx']
        threshold = self.dynamic_params['param_adx_threshold']
 
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
        lambda col: ta.roc(close=col, length=self.dynamic_params['lookback_roc'])
        )
        return roc_panel.iloc[-1]

    def _calc_factor_x5_vol_breakout(self) -> pd.Series:
        vol5 = self.panel_volume.rolling(window=self.dynamic_params['lookback_ma_fast']).mean()
        vol60 = self.panel_volume.rolling(window=self.dynamic_params['lookback_ma_slow']).mean()
        ratio = vol5.iloc[-1] / vol60.iloc[-1]
        return ratio.replace([np.inf, -np.inf], np.nan) # 处理分母为0的情况

    def _calc_factor_x6_new_high_momentum(self) -> pd.Series:
        high60 = self.panel_hfq_close.rolling(window=self.dynamic_params['lookback_new_high']).max()
        ratio = self.panel_hfq_close.iloc[-1] / high60.iloc[-1]
        return ratio

    def _calc_factor_x7_volatility(self) -> pd.Series:
        returns = self.panel_hfq_close.pct_change(fill_method=None)
        volatility = returns.rolling(window=self.dynamic_params['lookback_volatility']).std()
        return volatility.iloc[-1]

    def _calc_factor_x8_liquidity(self) -> pd.Series:
        liquidity = self.panel_turnover.rolling(window=self.dynamic_params['lookback_liquidity']).mean()
        return liquidity.iloc[-1]

    # endregion
    
    # region: --- 4. 辅助方法 (基本与旧版兼容) ---

    def _log_to_db(self, level, message):
        return
        #SystemLog.objects.create(log_level=level, module_name=MODULE_NAME, message=message)

    def _load_dynamic_parameters_and_defs(self):
        logger.debug("加载动态策略参数和因子定义...")
        #params_qs = StrategyParameters.objects.filter(param_name__startswith='dynamic_')
        params_qs = StrategyParameters.objects.all()
        self.dynamic_params = {p.param_name: float(p.param_value) for p in params_qs}
        for p in params_qs:
            if p.param_name.startswith('lookback_') or p.param_name in ['param_top_n', 'param_adx_threshold']:
                self.dynamic_params[p.param_name] = int(p.param_value)
            else:
                self.dynamic_params[p.param_name] = float(p.param_value)
        #defs_qs = FactorDefinitions.objects.filter(is_active=True, factor_code__startswith='dynamic_')
        defs_qs = FactorDefinitions.objects.all()
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
        # atr14 = self.panel_close.apply(
        #     lambda s: ta.atr(
        #         high=self.panel_high[s.name].astype(float), 
        #         low=self.panel_low[s.name].astype(float), 
        #         close=s.astype(float),  # 这里的 's' 现在是正确的不复权收盘价序列
        #         length=14
        #     ).iloc[-1]
        # )
        # 1. 准备基础数据面板
        high = self.panel_high
        low = self.panel_low
        close = self.panel_close
        
        # 2. 计算前一日收盘价 (整个面板一次性操作)
        prev_close = close.shift(1)
        
        # 3. 向量化计算TR的三个组成部分
        range1 = high - low
        range2 = (high - prev_close).abs()
        range3 = (low - prev_close).abs()
        
        # 4. 向量化计算真实波幅 (True Range)
        #    使用 np.maximum 逐元素比较，找出三者中的最大值。这比多次concat+max更高效。
        true_range = np.maximum(range1, range2)
        true_range = np.maximum(true_range, range3)
        
        # 5. 使用ewm(指数加权移动)计算ATR，这等同于技术分析中的Wilder's Smoothing
        #    - alpha = 1/length 是Wilder平滑的定义
        #    - adjust=False 使用递归平滑公式，与标准技术指标库行为一致
        #    - min_periods=14 确保有足够数据才开始计算
        atr_panel = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # 6. 获取最后一天的ATR值，得到一个Series，其结构与原代码输出完全一致
        atr14 = atr_panel.iloc[-1]
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
                'OLD': self.dynamic_weights['OLD'] * top_dimension_scores.loc[stock_code, 'Score_OLD'],
            }
            
            # 步骤2: 提取正向贡献值 (PDCV) 并计算策略DNA权重 (SSW)
            # ... (此部分逻辑保持不变)
            pdcv = {k: max(0, v) for k, v in dcv.items()}
            total_pdcv = sum(pdcv.values())
            
            if total_pdcv <= 1e-9:
                logger.warning(f"股票 {stock_code} 信号混乱 (Total_PDCV <= 0)，采用保守的QD策略作为其开盘区间。")
                ssw = {'MT': 0, 'BO': 0, 'MR': 0, 'QD': 1.0, 'OLD': 0}
            else:
                ssw = {k: v / total_pdcv for k, v in pdcv.items()}
            strategy_dna_str = "|".join([f"{k}:{v:.2f}" for k, v in ssw.items()])
            # 步骤3: 加权合成动态k值
            # ... (此部分逻辑保持不变)
            k_gap_dynamic = (
                ssw['MT'] * p['dynamic_miopmaop_k_gap_base_mt'] +
                ssw['BO'] * p['dynamic_miopmaop_k_gap_base_bo'] +
                ssw['MR'] * p['dynamic_miopmaop_k_gap_base_mr'] +
                ssw['QD'] * p['dynamic_miopmaop_k_gap_base_qd'] +
                ssw['OLD'] * p['k_gap']
            )
            k_drop_dynamic = (
                ssw['MT'] * p['dynamic_miopmaop_k_drop_base_mt'] +
                ssw['BO'] * p['dynamic_miopmaop_k_drop_base_bo'] +
                ssw['MR'] * p['dynamic_miopmaop_k_drop_base_mr'] +
                ssw['QD'] * p['dynamic_miopmaop_k_drop_base_qd'] +
                ssw['OLD'] * p['k_drop']
            )
            k_gap_dynamic=p['k_gap']
            k_drop_dynamic=p['k_drop']
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
                        f"SSW(MT/BO/MR/QD/OLD):({ssw['MT']:.2f}/{ssw['BO']:.2f}/{ssw['MR']:.2f}/{ssw['QD']:.2f}/{ssw['OLD']:.2f}), "
                        f"k_gap_dyn:{k_gap_dynamic:.2f}, k_drop_dyn:{k_drop_dynamic:.2f}, MIOP:{miop:.2f}, MAOP:{maop:.2f}")
 
        if not plans:
            return pd.DataFrame()
 
        plan_df = pd.DataFrame(plans).dropna(subset=['miop', 'maop'])
        logger.debug("动态交易预案生成完成。")
        return plan_df

    @transaction.atomic
    def _save_results(self, raw_factors_df, norm_scores_df, trading_plan_df):
        logger.debug("开始将结果保存到数据库...")
        from django.db import connection
        logger.debug("开始将结果高速保存到数据库 (临时禁用触发器)...")
        
        factor_table = DailyFactorValues._meta.db_table
        plan_table = DailyTradingPlan._meta.db_table
        with connection.cursor() as cursor:
            try:
                # 1. 临时禁用目标表的触发器 (包括外键约束)
                logger.debug(f"禁用表 {factor_table} 和 {plan_table} 的触发器...")
                cursor.execute(f'ALTER TABLE "{factor_table}" DISABLE TRIGGER ALL;')
                cursor.execute(f'ALTER TABLE "{plan_table}" DISABLE TRIGGER ALL;')
                # 2. 准备并写入每日因子值
                # 优化：先删除当日旧数据，再用纯粹的 bulk_create，比 ignore_conflicts 更快
                DailyFactorValues.objects.filter(
                    trade_date=self.trade_date
                ).exclude(
                    stock_code_id=MARKET_INDICATOR_CODE
                ).delete()
                
                factor_values_to_create = []
                for stock_code, row in raw_factors_df.iterrows():
                    for factor_code, raw_value in row.items():
                        norm_score = norm_scores_df.loc[stock_code, factor_code]
                        factor_values_to_create.append(
                            DailyFactorValues(
                                stock_code_id=stock_code, trade_date=self.trade_date,
                                factor_code_id=factor_code, raw_value=min(Decimal(str(raw_value)),Decimal(999999999)),
                                norm_score=Decimal(str(norm_score))
                            )
                        )
                # 优化：增加 batch_size 进一步提升性能和降低内存占用
                DailyFactorValues.objects.bulk_create(factor_values_to_create, batch_size=1000)
                logger.debug(f"已高速写入 {len(factor_values_to_create)} 条因子数据。")
                # 3. 准备并写入每日交易预案
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
                DailyTradingPlan.objects.bulk_create(plans_to_create, batch_size=1000)
                logger.debug(f"已高速写入 {len(plans_to_create)} 条交易预案。")
            except Exception as e:
                logger.critical(f"数据库写入事务发生根本性错误，即将回滚。根本原因: {e}", exc_info=True)
            finally:
                # 4. 无论成功与否，都必须重新启用触发器！
                logger.debug(f"重新启用表 {factor_table} 和 {plan_table} 的触发器...")
                cursor.execute(f'ALTER TABLE "{factor_table}" ENABLE TRIGGER ALL;')
                cursor.execute(f'ALTER TABLE "{plan_table}" ENABLE TRIGGER ALL;')
        
        # # 1. 保存每日因子值
        # factor_values_to_create = []
        # for stock_code, row in raw_factors_df.iterrows():
        #     for factor_code, raw_value in row.items():
        #         norm_score = norm_scores_df.loc[stock_code, factor_code]
        #         factor_values_to_create.append(
        #             DailyFactorValues(
        #                 stock_code_id=stock_code, trade_date=self.trade_date,
        #                 factor_code_id=factor_code, raw_value=Decimal(str(raw_value)),
        #                 norm_score=Decimal(str(norm_score))
        #             )
        #         )
        # DailyFactorValues.objects.bulk_create(factor_values_to_create, ignore_conflicts=True)

        # # 2. 保存每日交易预案
        # plan_date = self.trade_date + timedelta(days=1)
        # DailyTradingPlan.objects.filter(plan_date=plan_date).delete()
        
        # plans_to_create = []
        # for _, row in trading_plan_df.iterrows():
        #     plans_to_create.append(
        #         DailyTradingPlan(
        #             plan_date=plan_date, stock_code_id=row['stock_code'],
        #             rank=row['rank'], final_score=Decimal(str(row['final_score'])),
        #             miop=Decimal(str(row['miop'])).quantize(Decimal('0.01')),
        #             maop=Decimal(str(row['maop'])).quantize(Decimal('0.01')),
        #             status=DailyTradingPlan.StatusChoices.PENDING,
        #             strategy_dna=row['strategy_dna']
        #         )
        #     )
        # DailyTradingPlan.objects.bulk_create(plans_to_create)
        
        log_message = f"T-1日({self.trade_date})动态选股完成, T日({plan_date})预案如下:\n"
        log_message += trading_plan_df.to_string(index=False)
        self._log_to_db('INFO', log_message)
        logger.debug("所有结果已成功保存到数据库。")

    # endregion
