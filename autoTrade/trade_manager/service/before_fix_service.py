# trade_manager/service/before_fix_service.py

import logging
from datetime import date, timedelta, datetime
from decimal import Decimal, ROUND_HALF_UP
from django.utils import timezone
from django.db import transaction
from django.db.models import Q

# 导入项目内的模型
from common.models import (
    CorporateAction,
    DailyTradingPlan,
    Position,
    DailyQuotes,
    SystemLog
)

# 配置日志记录器
logger = logging.getLogger(__name__)


class BeforeFixService:
    """
    T日开盘前校准与预案修正服务。

    职责:
    1. 检查当天是否已成功执行过，防止重复运行。
    2. 获取T日的除权除息事件。
    3. 计算受影响股票的价格调整比率。
    4. 根据比率修正“每日交易预案”中的MIOP和MAOP。
    5. 根据比率修正“持仓信息”中的止盈止损价。
    6. 对近期发生配股的股票进行特殊风险处理。
    """
    MODULE_NAME = '盘前校准与预案修正'
    # 可配置参数
    MAX_PLAN_LOOKBACK_DAYS = 14  # 查找交易预案的最大回溯天数
    RIGHTS_ISSUE_LOOKBACK_DAYS = 30 # 配股事件特殊处理的回溯交易日数

    def __init__(self, execution_date: date = None):
        """
        初始化服务。
        :param execution_date: T日，即执行校准的日期。如果为None，则默认为当天。
        """
        self.t_day = execution_date if execution_date else date.today()
        self.t_minus_1_day = None
        self.adjustment_ratios = {} # 存储 {stock_code: ratio}
        logger.debug(f"[{self.MODULE_NAME}] 服务初始化，目标T日: {self.t_day}")

    def _log_to_db(self, level: str, message: str):
        """辅助方法：将日志写入数据库"""
        try:
            SystemLog.objects.create(
                log_level=level,
                module_name=self.MODULE_NAME,
                message=message
            )
        except Exception as e:
            logger.error(f"无法将日志写入数据库: {e}")

    def _is_trading_day(self, check_date: date) -> bool:
        """检查指定日期是否为交易日"""
        is_trade_day = DailyQuotes.objects.filter(trade_date=check_date).exists()
        logger.info(f"检查日期 {check_date} 是否为交易日: {'是' if is_trade_day else '否'}")
        return is_trade_day

    def _get_last_trading_day(self, from_date: date) -> date | None:
        """获取指定日期之前的最后一个交易日"""
        last_day = DailyQuotes.objects.filter(
            trade_date__lt=from_date
        ).order_by('-trade_date').values_list('trade_date', flat=True).first()
        
        if last_day:
            logger.info(f"{from_date} 的前一个交易日 (T-1) 是: {last_day}")
        else:
            logger.warning(f"无法找到 {from_date} 的前一个交易日。")
        return last_day

    def _find_latest_pending_plan_date(self) -> date | None:
        """从T日开始向前回溯，查找最新的一个包含待执行预案的日期"""
        for i in range(self.MAX_PLAN_LOOKBACK_DAYS):
            check_date = self.t_day - timedelta(days=i)
            if DailyTradingPlan.objects.filter(
                plan_date=check_date,
                status=DailyTradingPlan.StatusChoices.PENDING
            ).exists():
                logger.info(f"找到待执行的交易预案，预案日期为: {check_date}")
                return check_date
        logger.warning(f"在过去 {self.MAX_PLAN_LOOKBACK_DAYS} 天内未找到任何待执行的交易预案。")
        return None

    def _calculate_adjusted_price(self, t_minus_1_close: Decimal, events: list[CorporateAction]) -> Decimal:
        """
        核心算法：根据事件列表计算除权除息参考价。
        处理顺序：1.除息 -> 2.送/转股 -> 3.配股
        """
        adjusted_price = t_minus_1_close
        
        # 按事件类型优先级排序
        event_priority = {
            CorporateAction.EventType.DIVIDEND: 1,
            CorporateAction.EventType.BONUS: 2,
            CorporateAction.EventType.TRANSFER: 2,
            CorporateAction.EventType.SPLIT: 2,
            CorporateAction.EventType.RIGHTS: 3,
        }
        sorted_events = sorted(events, key=lambda e: event_priority.get(e.event_type, 99))

        for event in sorted_events:
            # 1. 现金分红 (除息)
            if event.event_type == CorporateAction.EventType.DIVIDEND and event.dividend_per_share:
                adjusted_price -= event.dividend_per_share
            
            # 2. 送股/转增股/并股/拆股 (除权)
            elif event.event_type in [CorporateAction.EventType.BONUS, CorporateAction.EventType.TRANSFER, CorporateAction.EventType.SPLIT]:
                if event.shares_before and event.shares_after and event.shares_after > 0:
                    adjusted_price = adjusted_price * (event.shares_before / event.shares_after)

            # 3. 配股 (除权) - 注意：按需求，此计算结果不用于常规校准，但逻辑保留
            elif event.event_type == CorporateAction.EventType.RIGHTS:
                if event.shares_before and event.shares_after and event.rights_issue_price is not None and event.shares_after > 0:
                    adjusted_price = (event.shares_before * adjusted_price + (event.shares_after - event.shares_before) * event.rights_issue_price) / event.shares_after
        
        return adjusted_price

    @transaction.atomic
    def run(self):
        """执行盘前校准与修正的主流程"""

        self.t_minus_1_day = self._get_last_trading_day(self.t_day)
        if not self.t_minus_1_day:
            logger.error(f"无法确定T-1日，任务终止。")
            return

        # a. 获取T日所有除权除息信息
        events_on_t_day = CorporateAction.objects.filter(ex_dividend_date=self.t_day)
        if not events_on_t_day.exists():
            logger.debug(f"T日 ({self.t_day}) 无除权除息事件，无需校准。")
            return

        # 按股票代码分组事件
        events_by_stock = {}
        for event in events_on_t_day:
            events_by_stock.setdefault(event.stock_code, []).append(event)
        
        affected_codes = list(events_by_stock.keys())
        logger.info(f"T日共有 {len(affected_codes)} 只股票发生股权事件。")

        # 获取这些股票在T-1日的收盘价
        quotes_qs = DailyQuotes.objects.filter(
            trade_date=self.t_minus_1_day,
            stock_code_id__in=affected_codes
        )
        # 使用字典推导式构建我们需要的映射关系
        quotes_t_minus_1 = {quote.stock_code_id: quote for quote in quotes_qs}

        # b. 计算价格调整比率
        for stock_code, events in events_by_stock.items():
            if stock_code not in quotes_t_minus_1:
                logger.warning(f"股票 {stock_code} 在T-1日({self.t_minus_1_day})无行情数据（可能停牌），跳过校准。")
                continue
            
            close_t_minus_1 = quotes_t_minus_1[stock_code].close
            if close_t_minus_1 <= 0:
                logger.warning(f"股票 {stock_code} 在T-1日收盘价为0或负数，不合理，跳过校准。")
                continue

            adjusted_close = self._calculate_adjusted_price(close_t_minus_1, events)
            ratio = adjusted_close / close_t_minus_1
            self.adjustment_ratios[stock_code] = ratio
            logger.info(f"股票 {stock_code}: T-1收盘价={close_t_minus_1}, 校准后价格={adjusted_close:.2f}, 调整比率={ratio:.6f}")

        # c. 修正交易预案
        self._process_trading_plans()

        # d. 修正持仓风控
        self._process_positions()

        # e. 配股事件特殊处理
        self._handle_rights_issue_special_case()

        logger.info(f"[{self.MODULE_NAME}] 任务成功完成。共处理 {len(self.adjustment_ratios)} 只股票的常规校准。")

    def _process_trading_plans(self):
        """修正交易预案中的MIOP和MAOP"""
        plan_date_to_fix = self._find_latest_pending_plan_date()
        if not plan_date_to_fix:
            return

        plans_to_fix = DailyTradingPlan.objects.filter(
            plan_date=plan_date_to_fix,
            status=DailyTradingPlan.StatusChoices.PENDING,
            stock_code__in=self.adjustment_ratios.keys()
        )

        plans_to_update = []
        for plan in plans_to_fix:
            ratio = self.adjustment_ratios[plan.stock_code_id]
            original_miop = plan.miop
            original_maop = plan.maop
            
            plan.miop = (original_miop * Decimal(str(ratio))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            plan.maop = (original_maop * Decimal(str(ratio))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            plans_to_update.append(plan)
            logger.info(f"交易预案修正: {plan.stock_code}, MIOP: {original_miop}->{plan.miop}, MAOP: {original_maop}->{plan.maop}")

        if plans_to_update:
            DailyTradingPlan.objects.bulk_update(plans_to_update, ['miop', 'maop'])
            logger.info(f"成功批量更新 {len(plans_to_update)} 条交易预案。")

    def _process_positions(self):
        """修正持仓中的止盈止损价"""
        positions_to_fix = Position.objects.filter(
            status=Position.StatusChoices.OPEN,
            stock_code__in=self.adjustment_ratios.keys()
        )

        positions_to_update = []
        for pos in positions_to_fix:
            ratio = self.adjustment_ratios[pos.stock_code_id]
            original_sl = pos.current_stop_loss
            original_tp = pos.current_take_profit

            pos.current_stop_loss = (original_sl * Decimal(str(ratio))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            pos.current_take_profit = (original_tp * Decimal(str(ratio))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            positions_to_update.append(pos)
            logger.info(f"持仓风控修正: {pos.stock_code}, 止损: {original_sl}->{pos.current_stop_loss}, 止盈: {original_tp}->{pos.current_take_profit}")

        if positions_to_update:
            Position.objects.bulk_update(positions_to_update, ['current_stop_loss', 'current_take_profit'])
            logger.info(f"成功批量更新 {len(positions_to_update)} 条持仓记录。")

    def _handle_rights_issue_special_case(self):
        """处理30个交易日内有配股事件的股票"""
        # 1. 获取过去30个交易日的日期列表
        recent_trading_days = list(
            DailyQuotes.objects.filter(trade_date__lte=self.t_day)
            .order_by('-trade_date')
            .values_list('trade_date', flat=True)[:self.RIGHTS_ISSUE_LOOKBACK_DAYS]
        )
        if not recent_trading_days:
            logger.warning("无法获取最近交易日列表，跳过配股特殊处理。")
            return

        # 2. 查找在此期间发生配股的股票
        rights_issue_stocks = list(
            CorporateAction.objects.filter(
                event_type=CorporateAction.EventType.RIGHTS,
                ex_dividend_date__in=recent_trading_days
            ).values_list('stock_code', flat=True).distinct()
        )
        if not rights_issue_stocks:
            logger.info("近期无配股事件，无需特殊处理。")
            return
        
        logger.warning(f"检测到 {len(rights_issue_stocks)} 只股票近期有配股事件: {rights_issue_stocks}，将进行风险剔除。")

        # 3. 处理交易预案
        plan_date_to_fix = self._find_latest_pending_plan_date()
        if plan_date_to_fix:
            plans_to_void = DailyTradingPlan.objects.filter(
                plan_date=plan_date_to_fix,
                status=DailyTradingPlan.StatusChoices.PENDING,
                stock_code__in=rights_issue_stocks
            )
            plans_to_update = []
            for plan in plans_to_void:
                plan.miop = Decimal('99999.00')
                plan.maop = Decimal('0.00')
                plans_to_update.append(plan)
            
            if plans_to_update:
                DailyTradingPlan.objects.bulk_update(plans_to_update, ['miop', 'maop'])
                logger.info(f"配股风险处理：将 {len(plans_to_update)} 条交易预案的MIOP/MAOP置为无效。")

        # 4. 处理持仓
        positions_to_void = Position.objects.filter(
            status=Position.StatusChoices.OPEN,
            stock_code__in=rights_issue_stocks
        )
        positions_to_update = []
        for pos in positions_to_void:
            pos.current_take_profit = Decimal('0.00')
            pos.current_stop_loss = Decimal('99999.00')
            positions_to_update.append(pos)
        
        if positions_to_update:
            Position.objects.bulk_update(positions_to_update, ['current_take_profit', 'current_stop_loss'])
            logger.info(f"配股风险处理：将 {len(positions_to_update)} 条持仓的止盈/止损置为紧急退出状态。")


# --- 如何在项目中使用这个服务 ---
# 你可以在一个Django Management Command或者定时任务（如Celery）中调用它
#
# from trade_manager.service.before_fix_service import BeforeFixService
#
# def run_daily_premarket_fix():
#     # 默认使用当天日期
#     service = BeforeFixService()
#     service.run()
#
# def run_backtest_premarket_fix(some_date):
#     # 传入指定日期进行回测
#     service = BeforeFixService(execution_date=some_date)
#     service.run()

