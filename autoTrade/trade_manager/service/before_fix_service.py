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
    SystemLog,
    DailyFactorValues
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

    def _calculate_adjustment_ratios(self, events: list[CorporateAction]) -> tuple[Decimal, Decimal]:
        """
        核心算法：根据事件列表计算除权除息参考价。
        处理顺序：1.除息 -> 2.送/转股 -> 3.配股
        """
        price_ratio = Decimal('1.0')
        quantity_ratio = Decimal('1.0')


        
        # 按事件类型优先级排序
        event_priority = {
            CorporateAction.EventType.DIVIDEND: 1,
            CorporateAction.EventType.BONUS: 2,
            CorporateAction.EventType.TRANSFER: 2,
            CorporateAction.EventType.SPLIT: 2
            #CorporateAction.EventType.RIGHTS: 3,
        }
        sorted_events = sorted(events, key=lambda e: event_priority.get(e.event_type, 99))

        for event in sorted_events:
            # 送股/转增股/并股/拆股 (除权)
            if event.event_type in [CorporateAction.EventType.BONUS, CorporateAction.EventType.TRANSFER, CorporateAction.EventType.SPLIT]:
                if event.shares_before and event.shares_after and event.shares_after > 0:
                    # 价格比率 = 旧股数 / 新股数
                    price_ratio *= (event.shares_before / event.shares_after)
                    # 数量比率 = 新股数 / 旧股数
                    quantity_ratio *= (event.shares_after / event.shares_before)
        
        return price_ratio, quantity_ratio

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
            logger.debug(f"T日 ({self.t_day}) 无除权除息事件，无需校准除权，但继续执行评分风控调整。")
            # 无除权事件也要执行评分驱动的风控校准
            self.apply_score_based_risk_adjustments()
            return

        # 按股票代码分组事件
        events_by_stock = {}
        for event in events_on_t_day:
            if event.event_type != CorporateAction.EventType.RIGHTS:
                 events_by_stock.setdefault(event.stock_code, []).append(event)
        
        affected_codes = list(events_by_stock.keys())
        if not affected_codes:
            logger.info("T日只有配股事件，常规校准流程跳过。")
            self._handle_rights_issue_special_case() # 仍然要处理配股的特殊情况
            # 无常规校准时，仍继续执行评分风控调整
            self.apply_score_based_risk_adjustments()
            return
        logger.info(f"T日共有 {len(affected_codes)} 只股票发生股权事件。")

        # 获取这些股票在T-1日的收盘价
        quotes_qs = DailyQuotes.objects.filter(
            trade_date=self.t_minus_1_day,
            stock_code_id__in=affected_codes
        )
        # 使用字典推导式构建我们需要的映射关系
        quotes_t_minus_1 = {quote.stock_code_id: quote for quote in quotes_qs}
        self.adjustment_ratios = {} 
        # b. 计算价格调整比率
        for stock_code, events in events_by_stock.items():
            if stock_code not in quotes_t_minus_1:
                logger.warning(f"股票 {stock_code} 在T-1日({self.t_minus_1_day})无行情数据（可能停牌），跳过校准。")
                continue
            
            close_t_minus_1 = quotes_t_minus_1[stock_code].close
            if close_t_minus_1 <= 0:
                logger.warning(f"股票 {stock_code} 在T-1日收盘价为0或负数，不合理，跳过校准。")
                continue
            # 1. 计算送/转/拆/并股的比率
            price_ratio_st, quantity_ratio_st = self._calculate_adjustment_ratios(events)
            # 2. 计算分红的价格影响
            total_dividend = sum(e.dividend_per_share for e in events if e.event_type == CorporateAction.EventType.DIVIDEND and e.dividend_per_share)
            
            price_ratio_div = Decimal('1.0')
            if total_dividend > 0:
                # 分红的价格调整比率 = (收盘价 - 分红) / 收盘价
                price_ratio_div = (close_t_minus_1 - total_dividend) / close_t_minus_1
            # 3. 合并总比率
            final_price_ratio = price_ratio_st * price_ratio_div
            final_quantity_ratio = quantity_ratio_st # 分红不影响数量
            self.adjustment_ratios[stock_code] = (final_price_ratio, final_quantity_ratio)
            
            logger.info(f"股票 {stock_code}: T-1收盘价={close_t_minus_1}, "
                        f"价格调整比率={final_price_ratio:.6f}, "
                        f"数量调整比率={final_quantity_ratio:.6f}")

        # c. 修正交易预案
        self._process_trading_plans()

        # d. 修正持仓风控
        self._process_positions()

        # e. 配股事件特殊处理
        self._handle_rights_issue_special_case()

        # f. 无论是否存在除权事件，均执行评分驱动的风控校准
        self.apply_score_based_risk_adjustments()

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
            price_ratio, _ = self.adjustment_ratios[plan.stock_code_id]
            original_miop = plan.miop
            original_maop = plan.maop
            
            plan.miop = (original_miop * price_ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            plan.maop = (original_maop * price_ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            plans_to_update.append(plan)
            logger.info(f"交易预案修正: {plan.stock_code_id}, MIOP: {original_miop}->{plan.miop}, MAOP: {original_maop}->{plan.maop}")

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
        update_fields = ['entry_price', 'quantity', 'current_stop_loss', 'current_take_profit']
        for pos in positions_to_fix:
            price_ratio, quantity_ratio = self.adjustment_ratios[pos.stock_code_id]
            
            original_ep = pos.entry_price
            original_qty = pos.quantity
            original_sl = pos.current_stop_loss
            original_tp = pos.current_take_profit
            # 【新增】调整成本价
            pos.entry_price = (original_ep * price_ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # 【新增】调整持仓数量，并取整到股
            pos.quantity = int((Decimal(str(original_qty)) * quantity_ratio).to_integral_value(rounding=ROUND_HALF_UP))
            # 【修改】调整止盈止损价
            pos.current_stop_loss = (original_sl * price_ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            pos.current_take_profit = (original_tp * price_ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            positions_to_update.append(pos)
            logger.info(f"持仓风控修正: {pos.stock_code_id}, "
                        f"成本价: {original_ep:.2f} -> {pos.entry_price:.2f}, "
                        f"数量: {original_qty} -> {pos.quantity}, "
                        f"止损: {original_sl:.2f} -> {pos.current_stop_loss:.2f}, "
                        f"止盈: {original_tp:.2f} -> {pos.current_take_profit:.2f}")

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


    def backtest_handle_one_word_board(self):
        """
        【回测专用工具】检查T日交易预案中的股票是否为一字板，并将其设置为无法买入。
        此方法不应在实盘`run`流程中调用，仅供回测模块在`run()`之后调用。
        一字板定义：当天 open == high == low == close，且成交量大于0。
        """
        logger.info(f"[回测工具] 开始为日期 {self.t_day} 检查一字板股票...")
        # 1. 找到T日所有待执行的交易预案
        pending_plans = DailyTradingPlan.objects.filter(
            plan_date=self.t_day,
            status=DailyTradingPlan.StatusChoices.PENDING
        )
        if not pending_plans.exists():
            logger.info(f"[回测工具] 日期 {self.t_day} 没有待执行的交易预案，无需检查。")
            return
        stock_codes_in_plan = list(pending_plans.values_list('stock_code_id', flat=True))
        # 2. 获取这些股票在T日的行情数据
        quotes_on_t_day = DailyQuotes.objects.filter(
            trade_date=self.t_day,
            stock_code_id__in=stock_codes_in_plan
        )
        one_word_board_stocks = []
        for quote in quotes_on_t_day:
            # 3. 判断是否为一字板
            is_one_word_board = (
                quote.open == quote.high and
                quote.high == quote.low and
                quote.low == quote.close
            )
            if is_one_word_board:
                one_word_board_stocks.append(quote.stock_code_id)
        if not one_word_board_stocks:
            logger.info(f"[回测工具] 检查完毕，在交易预案中未发现一字板股票。")
            return
        logger.warning(
            f"[回测工具] 检测到 {len(one_word_board_stocks)} 只一字板股票: {one_word_board_stocks}。"
            f"将修改其交易预案以阻止回测买入。"
        )
        # 4. 对一字板股票的交易预案进行处理，使其无法被买入
        plans_to_update_qs = DailyTradingPlan.objects.filter(
            plan_date=self.t_day,
            stock_code_id__in=one_word_board_stocks,
            status=DailyTradingPlan.StatusChoices.PENDING
        )
        plans_to_update = []
        for plan in plans_to_update_qs:
            # 设置一个不可能达成的价格区间
            plan.miop = Decimal('99999.00')  # 最低价设为极高
            plan.maop = Decimal('0.00')      # 最高价设为极低
            plans_to_update.append(plan)
            logger.info(f"[回测工具] 已将股票 {plan.stock_code_id} 的预案设为无效，"
                        f"MIOP: {plan.miop}, MAOP: {plan.maop}")
        if plans_to_update:
            DailyTradingPlan.objects.bulk_update(plans_to_update, ['miop', 'maop'])
            logger.info(f"[回测工具] 成功更新 {len(plans_to_update)} 条交易预案，以模拟一字板无法买入。")

    def apply_score_based_risk_adjustments(self):
        """
        基于昨日选股结果(针对T日的预案)对当前持仓执行评分驱动的风控校准：
        - 若评分 < -0.2：直接“抛出”风控（止盈=0，止损=999999）。
        - 若 -0.2 <= 评分 < 0：将止盈/止损分别向成本价靠拢50%。
        说明：使用 `DailyTradingPlan(plan_date=T日)` 的 `final_score` 作为昨日选股后的评分。
        """
        try:
            open_positions = list(
                Position.objects.filter(status=Position.StatusChoices.OPEN)
            )
            if not open_positions:
                logger.info("评分风控调整：当前无持仓，跳过。")
                return

            stock_codes = [pos.stock_code_id for pos in open_positions]
            # 单次查询：使用 PostgreSQL DISTINCT ON 取每只股票在 T 日之前最近一次评分
            latest_score_rows = (
                DailyFactorValues.objects
                .filter(
                    factor_code_id='ML_STOCK_SCORE',
                    trade_date__lt=self.t_day,
                    stock_code_id__in=stock_codes
                )
                .order_by('stock_code_id', '-trade_date')
                .distinct('stock_code_id')
                .values('stock_code_id', 'norm_score', 'raw_value')
            )
            score_map = {}
            for row in latest_score_rows:
                val = row.get('norm_score')
                if val is None:
                    val = row.get('raw_value')
                score_map[row['stock_code_id']] = val

            if not score_map:
                logger.info("评分风控调整：无历史评分，跳过。")
                return

            updates = []
            for pos in open_positions:
                score = score_map.get(pos.stock_code_id)
                if score is None:
                    continue
                # 统一转为Decimal进行比较
                try:
                    score_dec = Decimal(str(score))
                except Exception:
                    continue

                entry = pos.entry_price
                d_tp = pos.current_take_profit - entry
                d_sl = pos.current_stop_loss - entry

                # 熔断：评分<-0.5 直接抛出
                if score_dec < Decimal('-0.5'):
                    pos.current_take_profit = Decimal('0.00')
                    pos.current_stop_loss = Decimal('999999.00')
                    updates.append(pos)
                    logger.warning(
                        f"评分风控调整：{pos.stock_code_id} 评分={score_dec} < -0.5，熔断触发，设置紧急退出风控(TP=0, SL=999999)。"
                    )
                # 负分：缩小幅度，比例 = max(0, 1 + 2*score)
                elif score_dec < Decimal('0'):
                    factor = (Decimal('1.0') + Decimal('2.0') * score_dec)
                    if factor < Decimal('0'):
                        factor = Decimal('0')
                    new_tp = (entry + d_tp * factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    new_sl = (entry + d_sl * factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    # 不变式守护
                    if new_sl >= new_tp:
                        # 轻量守护，尽量保持与entry的相对次序
                        new_sl = min(new_sl, entry - Decimal('0.01'))
                        new_tp = max(new_tp, entry + Decimal('0.01'))
                    if new_tp != pos.current_take_profit or new_sl != pos.current_stop_loss:
                        pos.current_take_profit = new_tp
                        pos.current_stop_loss = new_sl
                        updates.append(pos)
                        logger.info(
                            f"评分风控调整：{pos.stock_code_id} 评分={score_dec} < 0，按系数{factor:.3f}向成本价收敛 -> TP={new_tp}, SL={new_sl}。"
                        )
                # 奖励：评分>0.2 扩大幅度，放大 = 1 + min(0.5, (score-0.2)*2)
                elif score_dec > Decimal('0.2'):
                    expand = Decimal('1.0') + min(Decimal('0.5'), (score_dec - Decimal('0.2')) * Decimal('2.0'))
                    # 保障上限1.5
                    if expand > Decimal('1.5'):
                        expand = Decimal('1.5')
                    new_tp = (entry + d_tp * expand).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    new_sl = (entry + d_sl * expand).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    # 不变式守护
                    if new_sl >= new_tp:
                        new_sl = min(new_sl, entry - Decimal('0.01'))
                        new_tp = max(new_tp, entry + Decimal('0.01'))
                    if new_tp != pos.current_take_profit or new_sl != pos.current_stop_loss:
                        pos.current_take_profit = new_tp
                        pos.current_stop_loss = new_sl
                        updates.append(pos)
                        logger.info(
                            f"评分风控调整：{pos.stock_code_id} 评分={score_dec} > 0.2，按系数{expand:.3f}扩大区间 -> TP={new_tp}, SL={new_sl}。"
                        )

            if updates:
                Position.objects.bulk_update(updates, ['current_take_profit', 'current_stop_loss'])
                logger.info(f"评分风控调整：已更新 {len(updates)} 条持仓的风控价格。")
            else:
                logger.info("评分风控调整：无需更新任何持仓。")
        except Exception as e:
            logger.error(f"评分风控调整失败: {e}", exc_info=True)