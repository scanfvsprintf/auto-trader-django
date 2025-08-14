# ==============================================================================
# 文件 5/5: trade_manager/service/simulate_trade_handler.py (修改)
# 描述: 模拟交易处理器，集成操作日志记录。
# ==============================================================================
# trade_manager/service/simulate_trade_handler.py

import logging
from datetime import time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, timedelta, datetime
from django.db import transaction
from django.utils import timezone

from .trade_handler import ITradeHandler
from common.models import Position, TradeLog, DailyQuotes, StockInfo, DailyFactorValues
from common.models.backtest_logs import BacktestOperationLog # 新增导入
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE # 新增导入

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from .simulate_trade import SimulateTradeService

logger = logging.getLogger(__name__)

class SimulateTradeHandler(ITradeHandler):
    """
    模拟交易处理器 (SimulateTradeHandler) - 集成操作日志。
    """

    def __init__(self, service: 'SimulateTradeService'):
        self.service = service
        self.current_price_node: Literal['OPEN', 'LOW', 'HIGH', 'CLOSE'] = 'CLOSE'

    def get_opening_price(self, stock_code: str) -> Decimal:
        try:
            quote = DailyQuotes.objects.get(
                stock_code_id=stock_code,
                trade_date=self.service.current_date
            )
            return quote.open
        except DailyQuotes.DoesNotExist:
            logger.warning(f"[回测] 无法在 {self.service.current_date} 找到 {stock_code} 的行情数据，返回0。")
            return Decimal('0.00')

    def get_realtime_price(self, stock_code: str) -> Decimal | None:
        try:
            quote = DailyQuotes.objects.get(
                stock_code_id=stock_code,
                trade_date=self.service.current_date
            )
            if self.current_price_node == 'LOW':
                return quote.low
            elif self.current_price_node == 'HIGH':
                return quote.high
            else:
                return quote.close
        except DailyQuotes.DoesNotExist:
            return None

    def get_available_balance(self) -> Decimal:
        return self.service.cash_balance

    @transaction.atomic
    def place_buy_order(self, stock_code: str, price: Decimal, quantity: int) -> None:
        amount = price * quantity
        commission = max(amount * self.service.COMMISSION_RATE, self.service.MIN_COMMISSION)
        total_cost = amount + commission

        if self.service.cash_balance < total_cost:
            raise ValueError(f"资金不足！")

        self.service.cash_balance -= total_cost
        logger.info(f"[回测] 买入 {stock_code} {quantity}股 @{price:.2f}, 花费: {amount:.2f}, 现金余额: {self.service.cash_balance:.2f}")

        entry_time = time(9, 30, 1)
        entry_datetime = timezone.make_aware(timezone.datetime.combine(self.service.current_date, entry_time))

        new_position = Position.objects.create(
            stock_code_id=stock_code, entry_datetime=entry_datetime, entry_price=price,
            quantity=quantity, status=Position.StatusChoices.OPEN,
            current_stop_loss=Decimal('0.00'), current_take_profit=Decimal('0.00')
        )

        trade_log = TradeLog.objects.create(
            position=new_position, stock_code_id=stock_code, trade_datetime=entry_datetime,
            trade_type=TradeLog.TradeTypeChoices.BUY, order_type=TradeLog.OrderTypeChoices.LIMIT,
            price=price, quantity=quantity, commission=commission,
            reason=TradeLog.ReasonChoices.ENTRY, status=TradeLog.StatusChoices.FILLED
        )
        
        self.service.last_buy_trade_id = trade_log.trade_id

        

    @transaction.atomic
    def sell_stock_by_market_price(self, position: Position, reason: str) -> None:
        if reason == TradeLog.ReasonChoices.STOP_LOSS:
            trigger_price = position.current_stop_loss
        else: # TAKE_PROFIT
            trigger_price = position.current_take_profit

        sell_price = (trigger_price * (Decimal('1.0') - self.service.SELL_SLIPPAGE_RATE))
        sell_price = sell_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        amount = sell_price * position.quantity
        commission = max(amount * self.service.COMMISSION_RATE, self.service.MIN_COMMISSION)
        stamp_duty = amount * self.service.STAMP_DUTY_RATE
        net_income = amount - commission - stamp_duty

        self.service.cash_balance += net_income
        logger.info(f"[回测] 卖出 {position.stock_code_id} {position.quantity}股 @{sell_price:.2f}, 现金余额: {self.service.cash_balance:.2f}")

        position.status = Position.StatusChoices.CLOSED
        position.save()

        sell_time = time(14, 57, 0)
        sell_datetime = timezone.make_aware(timezone.datetime.combine(self.service.current_date, sell_time))

        TradeLog.objects.create(
            position=position, stock_code_id=position.stock_code_id, trade_datetime=sell_datetime,
            trade_type=TradeLog.TradeTypeChoices.SELL, order_type=TradeLog.OrderTypeChoices.MARKET,
            price=sell_price, quantity=position.quantity, commission=commission,
            stamp_duty=stamp_duty, reason=reason, status=TradeLog.StatusChoices.FILLED
        )
        
        # --- 新增: 记录卖出操作日志 ---
        self._record_sell_operation(position, sell_price, reason)
        
    def _get_t_minus_1_date(self) -> date:
        """安全地获取T-1交易日"""
        try:
            return DailyQuotes.objects.filter(trade_date__lt=self.service.current_date).latest('trade_date').trade_date
        except DailyQuotes.DoesNotExist:
            logger.warning(f"无法找到 {self.service.current_date} 的前一个交易日。")
            return self.service.current_date - timedelta(days=1)

    def _record_buy_operation(self, position: Position):
        t_minus_1 = self._get_t_minus_1_date()
        
        # 获取M值
        try:
            m_value_obj = DailyFactorValues.objects.get(
                stock_code_id=MARKET_INDICATOR_CODE,
                factor_code_id='dynamic_M_VALUE',
                trade_date=t_minus_1
            )
            m_value = m_value_obj.raw_value
        except DailyFactorValues.DoesNotExist:
            m_value = None
        
        # 获取因子得分
        factor_scores_qs = DailyFactorValues.objects.filter(
            stock_code_id=position.stock_code_id,
            trade_date=t_minus_1
        ).exclude(factor_code__factor_code__startswith='dynamic_M') # 排除M值本身
        
        scores_str = "|".join([
            f"{f.factor_code.factor_code}:{f.norm_score:.2f}" 
            for f in factor_scores_qs.select_related('factor_code')
        ])
        
        # 获取止盈止损率 (在调用此函数时，Position应已被更新)
        profit_rate = (position.current_take_profit / position.entry_price) - 1 if position.entry_price > 0 else 0
        loss_rate = 1 - (position.current_stop_loss / position.entry_price) if position.entry_price > 0 else 0

        BacktestOperationLog.objects.create(
            backtest_run_id=self.service.backtest_run_id,
            position_id_ref=position.position_id,
            stock_code=position.stock_code_id,
            stock_name=position.stock_code.stock_name,
            trade_date=self.service.current_date,
            direction=BacktestOperationLog.Direction.BUY,
            exit_reason=None,
            profit_rate=profit_rate,
            loss_rate=loss_rate,
            buy_date_m_value=m_value,
            factor_scores=scores_str,
            price=position.entry_price,
            quantity=position.quantity,
            amount=position.entry_price * position.quantity
        )
        logger.debug(f"已记录买入操作日志 for Position ID: {position.position_id}")
        
    def _record_sell_operation(self, position: Position, sell_price: Decimal, reason: str):
        # 反查买入记录
        try:
            buy_op = BacktestOperationLog.objects.get(
                backtest_run_id=self.service.backtest_run_id,
                position_id_ref=position.position_id,
                direction=BacktestOperationLog.Direction.BUY
            )
            m_value = buy_op.buy_date_m_value
            scores_str = buy_op.factor_scores
            profit_rate = buy_op.profit_rate
            loss_rate = buy_op.loss_rate
        except BacktestOperationLog.DoesNotExist:
            logger.error(f"严重错误：无法找到 Position ID {position.position_id} 对应的买入操作日志！")
            m_value, scores_str, profit_rate, loss_rate = None, "", None, None

        BacktestOperationLog.objects.create(
            backtest_run_id=self.service.backtest_run_id,
            position_id_ref=position.position_id,
            stock_code=position.stock_code_id,
            stock_name=position.stock_code.stock_name,
            trade_date=self.service.current_date,
            direction=BacktestOperationLog.Direction.SELL,
            exit_reason=reason,
            profit_rate=profit_rate,
            loss_rate=loss_rate,
            buy_date_m_value=m_value,
            factor_scores=scores_str,
            price=sell_price,
            quantity=position.quantity,
            amount=sell_price * position.quantity
        )
        logger.debug(f"已记录卖出操作日志 for Position ID: {position.position_id}")
