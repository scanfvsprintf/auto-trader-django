# trade_manager/service/decision_order_service.py
# 版本: 2.0 - M(t)驱动的动态风险管理
# 描述: 此版本重构了止盈止损计算逻辑，使其与动态选股策略的市场状态判断(M(t))保持一致。
#       解决了旧版在非趋势行情中止损价可能高于止盈价的严重逻辑问题。

import logging
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import pandas_ta as ta

from django.db import transaction
from django.utils import timezone

from common.models import (
    DailyTradingPlan,
    Position,
    TradeLog,
    StrategyParameters,
    DailyQuotes,
    SystemLog,
    DailyFactorValues  # 新增导入
)
from .trade_handler import ITradeHandler
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE # 新增导入

# 配置日志记录器
logger = logging.getLogger(__name__)

class DecisionOrderService:
    """
    开盘决策与下单模块 (V2.0 - 动态风险版)。

    该服务负责在T日开盘后的黄金时间内，根据预案、实际开盘价和账户状态，
    做出最终的买入决策，并执行下单。其核心特色是，在订单成交后，
    能够根据T-1日的市场状态M(t)，为新持仓计算动态的、自适应的止盈止损价。
    """
    MODULE_NAME = '开盘决策与下单(动态风险版)'
    MAX_PLAN_LOOKBACK_DAYS = 14

    def __init__(self, handler: ITradeHandler, execution_date: date = None):
        """
        初始化服务。

        :param handler: 一个实现了 ITradeHandler 接口的实例，用于与交易环境交互。
        :param execution_date: T日，即执行决策的日期。如果为None，则默认为当天。
        """
        if not isinstance(handler, ITradeHandler):
            raise TypeError("传入的 handler 必须是 ITradeHandler 的一个实例。")
      
        self.handler = handler
        self.execution_date = execution_date if execution_date else date.today()
        self.params = self._load_strategy_parameters()
      
        # 新增：用于存储当日动态计算结果的实例变量
        self.current_max_positions = 0
        self.final_nominal_principal = Decimal('0.0')
 
        # 【全新】调用新的初始化引擎
        self._initialize_position_sizing_engine()
      
        logger.debug(f"[{self.MODULE_NAME}] 服务初始化。执行T日: {self.execution_date}")
        logger.debug(f"策略参数加载成功: {len(self.params)}个")
        logger.debug(f"当日动态最大持仓数: {self.current_max_positions}")
        logger.debug(f"当日动态单位名义本金: {self.final_nominal_principal:.2f}")

    def _initialize_position_sizing_engine(self):
        """
        【全新方法】
        在服务初始化时，完成所有基于T-1日M(t)的仓位 sizing 计算。
        """
        try:
            # 1. 获取T-1交易日
            t_minus_1_date = DailyQuotes.objects.filter(trade_date__lt=self.execution_date).latest('trade_date').trade_date
        except DailyQuotes.DoesNotExist:
            logger.error(f"无法找到 {self.execution_date} 的前一个交易日，动态仓位管理无法启动，将使用默认值。")
            self.current_max_positions = self.params.get('MIN_POSITIONS_COUNT', 1)
            self.final_nominal_principal = Decimal('0.0') # 导致无法买入
            return
 
        # 2. 获取T-1日的市场状态M(t)
        market_regime_M = self._get_market_regime_M(t_minus_1_date)
        logger.info(f"获取到 T-1 ({t_minus_1_date}) 的 M(t) = {market_regime_M:.4f}")
 
        # 3. 计算当日动态最大持仓数
        self.current_max_positions = self._calculate_dynamic_max_positions(market_regime_M)
        
        # 4. 计算当日动态单位名义本金
        self.final_nominal_principal = self._calculate_dynamic_nominal_principal(market_regime_M, t_minus_1_date)
 
    def _get_market_regime_M(self, t_minus_1_date: date) -> Decimal:
        """
        【全新方法】
        从数据库获取指定日期的 M(t) 值。
        """
        try:
            m_value_record = DailyFactorValues.objects.get(
                stock_code_id=MARKET_INDICATOR_CODE,
                trade_date=t_minus_1_date,
                factor_code_id='dynamic_M_VALUE'
            )
            return m_value_record.raw_value
        except DailyFactorValues.DoesNotExist:
            logger.error(f"严重警告: 无法在 {t_minus_1_date} 找到市场状态M(t)值！将使用最保守的中性值0.0进行计算。")
            return Decimal('0.0')
 
    def _calculate_dynamic_max_positions(self, M_t: Decimal) -> int:
        """
        【全新方法】
        根据M(t)计算动态最大持仓数 Current_Max_Positions。
        """
        S_min_pos = self.params['RISK_ADJ_POS_FLOOR_PCT']
        
        # i. 计算总仓位数风险缩放因子 S_pos(M(t))
        S_pos = S_min_pos + (1 - S_min_pos) * (M_t + 1) / 2
        
        # ii. 计算理论最大仓位数
        base_max_pos = self.params['ORIGINAL_MAX_POSITIONS']
        theoretical_max = Decimal(base_max_pos) * S_pos
        
        # iii. 取整并应用下限
        min_pos_count = self.params['MIN_POSITIONS_COUNT']
        current_max_positions = max(min_pos_count, int(theoretical_max.to_integral_value(rounding='ROUND_FLOOR')))
        
        logger.debug(f"动态持仓数计算: S_pos={S_pos:.4f}, 理财持仓={theoretical_max:.2f}, 最终取整={current_max_positions}")
        return current_max_positions
 
    def _calculate_dynamic_nominal_principal(self, M_t: Decimal, t_minus_1_date: date) -> Decimal:
        """
        【全新方法】
        根据M(t)计算动态单位名义本金 Final_Nominal_Principal。
        """
        # i. 获取当前总资产
        cash_balance = self.handler.get_available_balance()
        positions_market_value = Decimal('0.0')
        open_positions = Position.objects.filter(status=Position.StatusChoices.OPEN)
        if open_positions.exists():
            for pos in open_positions:
                try:
                    quote = DailyQuotes.objects.get(stock_code_id=pos.stock_code_id, trade_date=t_minus_1_date)
                    positions_market_value += quote.close * pos.quantity
                except DailyQuotes.DoesNotExist:
                    positions_market_value += pos.entry_price * pos.quantity
        
        total_assets = cash_balance + positions_market_value
        logger.debug(f"总资产计算: 现金{cash_balance:.2f} + 持仓市值{positions_market_value:.2f} = {total_assets:.2f}")
 
        # ii. 计算基准单位名义本金
        base_max_pos = self.params['ORIGINAL_MAX_POSITIONS']
        if base_max_pos <= 0: return Decimal('0.0')
        baseline_unit_principal = total_assets / Decimal(base_max_pos)
        
        # iii. 计算单位名义本金风险缩放因子 S_cap(M(t))
        S_min_cap = self.params['RISK_ADJ_CAPITAL_FLOOR_PCT']
        S_cap = S_min_cap + (1 - S_min_cap) * (M_t + 1) / 2
        S_cap=1
        # iv. 计算动态调整后的名义本金
        adjusted_unit_principal = baseline_unit_principal * S_cap
        
        logger.debug(f"动态名义本金计算: 基准本金={baseline_unit_principal:.2f}, S_cap={S_cap:.4f}, 调整后本金={adjusted_unit_principal:.2f}")
        
        # v. 确定最终下单名义本金 - 注意：与可用现金的比较将在下单时进行
        return adjusted_unit_principal

    def _log_to_db(self, level: str, message: str):
        """辅助方法：将日志写入数据库"""
        # 在高频回测中可以注释掉此方法以提高性能
        # SystemLog.objects.create(
        #     log_level=level,
        #     module_name=self.MODULE_NAME,
        #     message=message
        # )
        pass
    def _find_relevant_plan_date(self) -> date | None:
        # 1. 计算查询的起始日期
        start_date = self.execution_date - timedelta(days=self.MAX_PLAN_LOOKBACK_DAYS - 1)
        
        # 2. 执行一次数据库查询
        latest_plan = DailyTradingPlan.objects.filter(
            plan_date__gte=start_date,  # gte = greater than or equal to (大于等于)
            plan_date__lte=self.execution_date, # lte = less than or equal to (小于等于)
            status=DailyTradingPlan.StatusChoices.PENDING
        ).order_by('-plan_date').first() # 按日期降序排列，并取第一个
    
        # 3. 处理查询结果
        if latest_plan:
            found_date = latest_plan.plan_date
            if found_date != self.execution_date:
                logger.info(f"执行日 {self.execution_date} 无预案，回溯找到待执行预案，其生成日期为: {found_date}")
            else:
                logger.debug(f"找到当天 {found_date} 的待执行预案。")
            return found_date
        
        # 如果查询结果为空
        logger.warning(f"在过去 {self.MAX_PLAN_LOOKBACK_DAYS} 天内（从 {self.execution_date} 开始回溯）未找到任何待执行的交易预案。")
        return None
    def _load_strategy_parameters(self) -> dict:
        """从数据库加载所有策略参数到内存"""
        params = {}
        # 定义需要加载的参数及其默认值
        # 注意：这里的键名应与 initialize_strategy_parameters 中定义的完全一致
        required_params = {
            # 通用参数
            #'MAX_POSITIONS': '3',
            'MAX_CAPITAL_PER_POSITION': '20000.00',
            'k_slip': '0.002',
            'lookback_atr': '14',
            # 新版动态风险参数
            'risk_adj_tp_pct_min': '0.07',
            'risk_adj_tp_pct_max': '0.15',
            'risk_adj_sl_atr_min': '1.2',
            'risk_adj_sl_atr_max': '2.2',
            'risk_adj_max_loss_pct': '0.08',
            # 全新动态仓位参数
            'ORIGINAL_MAX_POSITIONS': '5',
            'MIN_POSITIONS_COUNT': '1',
            'RISK_ADJ_POS_FLOOR_PCT': '0.2',
            'RISK_ADJ_CAPITAL_FLOOR_PCT': '0.5',
        }
      
        db_params = {p.param_name: p.param_value for p in StrategyParameters.objects.all()}
      
        for key, default_value in required_params.items():
            value = db_params.get(key, Decimal(str(default_value)))
            if key in ['ORIGINAL_MAX_POSITIONS', 'MIN_POSITIONS_COUNT', 'lookback_atr']:
                params[key] = int(value)
            else:
                params[key] = Decimal(str(value))
        return params

    # --- 暴露给外部调度的核心函数 ---

    def adjust_trading_plan_daily(self):
        """
        函数一：执行每日交易预案再调整 (逻辑不变)。
        """
        logger.debug(f"开始执行 {self.execution_date} 的交易预案二次筛选...")
        relevant_plan_date = self._find_relevant_plan_date()
        if not relevant_plan_date:
            msg = f"在 {self.execution_date} 及之前 {self.MAX_PLAN_LOOKBACK_DAYS} 天内没有找到任何待执行的交易预案。"
            logger.debug(msg)
            self._log_to_db('WARNING', msg)
            return
        plans_today = DailyTradingPlan.objects.filter(
            plan_date=relevant_plan_date,
            status=DailyTradingPlan.StatusChoices.PENDING
        ).order_by('rank')

        if not plans_today.exists():
            msg = f"在 {self.execution_date} 没有找到待执行的交易预案。"
            logger.debug(msg)
            self._log_to_db('WARNING', msg)
            return

        plans_to_cancel = []
        for plan in plans_today:
            try:
                open_price = self.handler.get_opening_price(plan.stock_code_id)
                if open_price <= 0:
                    logger.warning(f"股票 {plan.stock_code_id} 开盘价为0或无效，视为不符合条件。")
                    plan.status = DailyTradingPlan.StatusChoices.CANCELLED
                    plans_to_cancel.append(plan)
                    continue

                if not (plan.miop <= open_price <= plan.maop):
                    msg = (f"预案 {plan.stock_code_id} (Rank:{plan.rank}) 开盘价 {open_price} "
                           f"不在区间 [{plan.miop}, {plan.maop}] 内，已作废。")
                    logger.debug(msg)
                    plan.status = DailyTradingPlan.StatusChoices.CANCELLED
                    plans_to_cancel.append(plan)

            except Exception as e:
                msg = f"获取 {plan.stock_code_id} 开盘价时发生错误: {e}，该预案作废。"
                logger.error(msg)
                self._log_to_db('ERROR', msg)
                plan.status = DailyTradingPlan.StatusChoices.CANCELLED
                plans_to_cancel.append(plan)

        if plans_to_cancel:
            with transaction.atomic():
                DailyTradingPlan.objects.bulk_update(plans_to_cancel, ['status'])
            logger.debug(f"成功作废 {len(plans_to_cancel)} 条不符合开盘条件的交易预案。")
        else:
            logger.debug("所有待执行预案均符合开盘价条件。")

    def execute_orders(self):
        """
        函数二：进行下单 (逻辑不变)。
        """
        logger.debug(f"开始执行 {self.execution_date} 的下单流程...")

        open_positions_count = Position.objects.filter(status=Position.StatusChoices.OPEN).count()
        # 使用动态计算的当日最大持仓数
        remaining_slots = self.current_max_positions - open_positions_count

        if remaining_slots <= 0:
            msg = f"当前持仓数 {open_positions_count} 已达或超过当日动态上限 {self.current_max_positions}，不进行买入。"
            logger.debug(msg)
            self._log_to_db('WARNING', msg)
            return

        relevant_plan_date = self._find_relevant_plan_date()
        if not relevant_plan_date:
            msg = f"在 {self.execution_date} 及之前 {self.MAX_PLAN_LOOKBACK_DAYS} 天内没有找到任何待执行的交易预案可供下单。"
            logger.debug(msg)
            self._log_to_db('INFO', msg)
            return

        candidates = DailyTradingPlan.objects.filter(
            plan_date=relevant_plan_date,
            status=DailyTradingPlan.StatusChoices.PENDING
        ).order_by('rank')
 
        if not candidates.exists():
            msg = f"在 {self.execution_date} 无符合条件的买入标的。"
            logger.debug(msg)
            self._log_to_db('INFO', msg)
            return

        for candidate in candidates:
            try:
                stock_code = candidate.stock_code_id
                open_price = self.handler.get_opening_price(stock_code)
              
                k_slip = self.params['k_slip']
                limit_price = (open_price * (Decimal('1.0') + k_slip)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
 
                # 获取下单时最终的名义本金
                available_balance = self.handler.get_available_balance()
                
                # self.final_nominal_principal 是已按M(t)调整过的值
                # 再结合硬性风控和流动性约束
                nominal_principal = min(
                    self.final_nominal_principal, 
                    self.params['MAX_CAPITAL_PER_POSITION'], 
                    available_balance
                )
                
                logger.debug(f"标的 {stock_code}: 动态调整后本金={self.final_nominal_principal:.2f}, "
                             f"单仓上限={self.params['MAX_CAPITAL_PER_POSITION']:.2f}, "
                             f"可用现金={available_balance:.2f}. "
                             f"最终名义本金={nominal_principal:.2f}")
 
                if limit_price <= 0:
                    logger.debug(f"标的 {stock_code}: 计算出的下单限价无效（{limit_price}），跳过。")
                    continue
 
                shares_to_buy = int(nominal_principal / limit_price)
                quantity = (shares_to_buy // 100) * 100
 
                if quantity < 100:
                    msg = (f"标的 {stock_code}: 计算出的名义本金 {nominal_principal:.2f} 不足以购买一手（100股）。")
                    logger.warning(msg)
                    self._log_to_db('WARNING', msg)
                    continue
 
                msg = (f"确定唯一买入标的: {candidate.stock_code.stock_name}({stock_code}) (Rank:{candidate.rank})。 "
                       f"计划以限价 {limit_price} 买入 {quantity} 股。")
                logger.info(msg)
                self._log_to_db('INFO', msg)
              
                self.handler.place_buy_order(stock_code, limit_price, quantity)
              
                candidate.status = DailyTradingPlan.StatusChoices.EXECUTED
                candidate.save()
 
                return
 
            except Exception as e:
                msg = f"处理候选股 {candidate.stock_code_id} 时发生严重错误: {e}"
                logger.error(msg, exc_info=True)
                self._log_to_db('CRITICAL', msg)
                continue
 
        logger.debug(f"已尝试所有 {len(candidates)} 个候选标的，均未成功买入。")

    def calculate_stop_profit_loss(self, trade_id: int):
        """
        函数三：止盈止损区间计算 (V2.0 重构版)。
        在订单成交后，为新持仓计算并更新由M(t)驱动的动态止盈止损价。
 
        :param trade_id: 已成交的买入交易在 tb_trade_log 中的唯一ID。
        """
        logger.debug(f"开始为 trade_id={trade_id} 计算动态止盈止损区间...")
        try:
            with transaction.atomic():
                # 1. 获取交易和持仓信息
                trade_log = TradeLog.objects.select_for_update().get(
                    trade_id=trade_id,
                    trade_type=TradeLog.TradeTypeChoices.BUY,
                    status=TradeLog.StatusChoices.FILLED
                )
                position = Position.objects.select_for_update().get(pk=trade_log.position_id)
 
                if position.current_stop_loss > 0 and position.current_take_profit > 0:
                    logger.warning(f"Position ID {position.position_id} 似乎已计算过止盈止损，将跳过。")
                    return
 
                stock_code = trade_log.stock_code_id
                aep = trade_log.price
                buy_date = trade_log.trade_datetime.date()
                t_minus_1_date = DailyQuotes.objects.filter(trade_date__lt=buy_date).latest('trade_date').trade_date
              
                # 2. 获取计算所需的核心数据：M(t) 和 ATR
                # 2.1 获取 T-1 日的市场状态 M(t)
                try:
                    m_value_record = DailyFactorValues.objects.get(
                        stock_code_id=MARKET_INDICATOR_CODE,
                        trade_date=t_minus_1_date,
                        factor_code_id='dynamic_M_VALUE'
                    )
                    market_regime_M = m_value_record.raw_value
                except DailyFactorValues.DoesNotExist:
                    logger.error(f"无法找到 {t_minus_1_date} 的市场状态M(t)值！将使用中性值0.0进行计算。")
                    market_regime_M = Decimal('0.0')

                # 2.2 获取计算 ATR 所需的历史行情
                lookback_days = self.params['lookback_atr'] + 50 # 增加buffer
                start_date_for_calc = t_minus_1_date - timedelta(days=lookback_days * 2)
 
                quotes_qs = DailyQuotes.objects.filter(
                    stock_code_id=stock_code,
                    trade_date__gte=start_date_for_calc,
                    trade_date__lte=t_minus_1_date
                ).order_by('trade_date')
 
                if len(quotes_qs) < self.params['lookback_atr']:
                    raise ValueError(f"股票 {stock_code} 在 {t_minus_1_date} 前的历史数据不足，无法计算ATR。")
 
                df = pd.DataFrame.from_records(quotes_qs.values('high', 'low', 'close'))
                df = df.astype(float)
 
                atr_series = ta.atr(df['high'], df['low'], df['close'], length=self.params['lookback_atr'])
                atr_14_buy = Decimal(str(atr_series.iloc[-1])) if not atr_series.empty else Decimal('0.0')

                # 3. 计算动态止盈价 g_new(y)
                tp_min = self.params['risk_adj_tp_pct_min']
                tp_max = self.params['risk_adj_tp_pct_max']
                tp_pct = tp_min + (tp_max - tp_min) * (market_regime_M + 1) / 2
                take_profit_price = aep * (1 + tp_pct)

                # 4. 计算自适应止损价 h_new(z)
                # 4.1 计算动态ATR乘数 k_h(M(t))
                kh_min = self.params['risk_adj_sl_atr_min']
                kh_max = self.params['risk_adj_sl_atr_max']
                k_h_dynamic = kh_min + (kh_max - kh_min) * (market_regime_M + 1) / 2
                
                # 4.2 计算动态波动止损线
                z1_dynamic_atr = aep - k_h_dynamic * atr_14_buy

                # 4.3 计算绝对最大亏损底线
                z2_max_loss = aep * (1 - self.params['risk_adj_max_loss_pct'])
              
                # 4.4 取最严格的止损位（价格最高者）
                stop_loss_price = max(z1_dynamic_atr, z2_max_loss)
              
                logger.debug(f"[{stock_code}] 止损线比较 (基于M(t)={market_regime_M:.4f}): "
                            f"动态ATR止损(乘数{k_h_dynamic:.2f})={z1_dynamic_atr:.2f}, "
                            f"绝对最大亏损={z2_max_loss:.2f}")
 
                # 5. 更新持仓信息表
                position.current_take_profit = take_profit_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                position.current_stop_loss = stop_loss_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                
                # 最终校验，防止出现极端情况
                if position.current_stop_loss >= position.current_take_profit:
                    logger.critical(f"严重逻辑错误！计算后止损价({position.current_stop_loss})仍高于或等于止盈价({position.current_take_profit})。将使用最大亏损底线作为止损。")
                    position.current_stop_loss = z2_max_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

                position.save(update_fields=['current_take_profit', 'current_stop_loss'])
                loss_pct = (aep - position.current_stop_loss) / aep if aep > 0 else Decimal('0.0')
                msg = (f"成功计算并更新 Position ID {position.position_id} ({stock_code}) 的动态风控价格: "
                       f"购入价={aep:.2f}, 止盈价={position.current_take_profit:.2f} (目标收益率 {tp_pct:.2%}), "
                       f"止损价={position.current_stop_loss:.2f} (最大容忍亏损 {loss_pct:.2%})")
                logger.info(msg)
                self._log_to_db('INFO', msg)
 
        except TradeLog.DoesNotExist:
            logger.error(f"Trade ID {trade_id} 不存在或不满足计算条件（非买入/未成交）。")
        except Position.DoesNotExist:
            logger.error(f"与 Trade ID {trade_id} 关联的 Position 不存在。")
        except Exception as e:
            msg = f"为 Trade ID {trade_id} 计算动态止盈止损时发生严重错误: {e}"
            logger.critical(msg, exc_info=True)
            self._log_to_db('CRITICAL', msg)
            raise

    # --- 工具函数 ---

    @staticmethod
    def initialize_strategy_parameters():
        """
        工具函数：初始化本模块所需的策略参数到数据库。
        这是一个幂等操作，可以安全地重复运行。
        """
        logger.info("开始初始化[开盘决策与下单模块-动态风险版]的策略参数...")

        params_to_define = {
            # 通用参数
            #'MAX_POSITIONS': {'value': '3', 'group': 'POSITION_MGMT', 'desc': '最大可具备的总仓位数'},
            'MAX_CAPITAL_PER_POSITION': {'value': '20000.00', 'group': 'POSITION_MGMT', 'desc': '每仓最大投入资金数(元)'},
            'k_slip': {'value': '0.002', 'group': 'ORDER_EXEC', 'desc': '下单滑点系数, 用于计算限价单价格'},
            'lookback_atr': {'value': '14', 'group': 'INDICATORS', 'desc': 'ATR计算周期'},
            
            # 新版 M(t) 驱动的动态风险参数
            'risk_adj_tp_pct_min': {'value': '0.07', 'group': 'RISK_ADJUSTED', 'desc': 'M(t)驱动-最小止盈目标百分比 (熊市)'},
            'risk_adj_tp_pct_max': {'value': '0.15', 'group': 'RISK_ADJUSTED', 'desc': 'M(t)驱动-最大止盈目标百分比 (牛市)'},
            'risk_adj_sl_atr_min': {'value': '1.2', 'group': 'RISK_ADJUSTED', 'desc': 'M(t)驱动-最小ATR止损乘数 (熊市)'},
            'risk_adj_sl_atr_max': {'value': '2.2', 'group': 'RISK_ADJUSTED', 'desc': 'M(t)驱动-最大ATR止损乘数 (牛市)'},
            'risk_adj_max_loss_pct': {'value': '0.08', 'group': 'RISK_ADJUSTED', 'desc': 'M(t)驱动-绝对最大亏损百分比'},
            # --- 全新动态仓位管理参数 ---
            'ORIGINAL_MAX_POSITIONS': {'value': '5', 'group': 'DYNAMIC_POS_MGMT', 'desc': '【动态仓位】策略基准最大持仓数'},
            'MIN_POSITIONS_COUNT': {'value': '1', 'group': 'DYNAMIC_POS_MGMT', 'desc': '【动态仓位】最小持仓数硬下限'},
            'RISK_ADJ_POS_FLOOR_PCT': {'value': '0.1', 'group': 'DYNAMIC_POS_MGMT', 'desc': '【动态仓位】总仓位数缩放因子的下限 S_min_pos (例如0.4代表最差情况持有基准的40%)'},
            'RISK_ADJ_CAPITAL_FLOOR_PCT': {'value': '0.6', 'group': 'DYNAMIC_POS_MGMT', 'desc': '【动态仓位】单位名义本金缩放因子的下限 S_min_cap (例如0.6代表最差情况投入基准的60%)'}
        }

        with transaction.atomic():
            for name, data in params_to_define.items():
                StrategyParameters.objects.update_or_create(
                    param_name=name,
                    defaults={
                        'param_value': Decimal(data['value']),
                        'group_name': data['group'],
                        'description': data['desc']
                    }
                )
      
        logger.info(f"成功初始化/更新 {len(params_to_define)} 个动态风险策略参数。")
