# trade_manager/service/simulate_trade_handler.py

import logging
from datetime import time
from decimal import Decimal, ROUND_HALF_UP

from django.db import transaction
from django.utils import timezone

from .trade_handler import ITradeHandler
from common.models import Position, TradeLog, DailyQuotes

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from .simulate_trade import SimulateTradeService

logger = logging.getLogger(__name__)

class SimulateTradeHandler(ITradeHandler):
    """
    模拟交易处理器 (SimulateTradeHandler)。
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
                logger.warning(f"[回测] 在非法的价格节点 {self.current_price_node} 调用了 get_realtime_price。")
                return quote.close
        except DailyQuotes.DoesNotExist:
            logger.warning(f"[回测] 无法在 {self.service.current_date} 找到 {stock_code} 的行情数据，返回 None。")
            return None

    def get_available_balance(self) -> Decimal:
        return self.service.cash_balance

    @transaction.atomic
    def place_buy_order(self, stock_code: str, price: Decimal, quantity: int) -> None:
        amount = price * quantity
        commission = max(amount * self.service.COMMISSION_RATE, self.service.MIN_COMMISSION)
        total_cost = amount + commission

        if self.service.cash_balance < total_cost:
            msg = f"[回测] 资金不足！尝试买入 {stock_code} 需 {total_cost:.2f}，但现金仅剩 {self.service.cash_balance:.2f}。"
            logger.error(msg)
            raise ValueError(msg)

        self.service.cash_balance -= total_cost
        logger.info(f"[回测] 买入 {stock_code} {quantity}股 @{price:.2f}, "
                    f"花费: {amount:.2f}, 佣金: {commission:.2f}, "
                    f"现金余额: {self.service.cash_balance:.2f}")

        # 细节优化：使用更真实的开盘时间
        entry_time = time(9, 30, 1)
        entry_datetime = timezone.make_aware(
            timezone.datetime.combine(self.service.current_date, entry_time)
        )

        new_position = Position.objects.create(
            stock_code_id=stock_code,
            entry_datetime=entry_datetime,
            entry_price=price,
            quantity=quantity,
            current_stop_loss=Decimal('0.00'),
            current_take_profit=Decimal('0.00'),
            status=Position.StatusChoices.OPEN
        )

        trade_log = TradeLog.objects.create(
            position=new_position,
            stock_code_id=stock_code,
            trade_datetime=entry_datetime,
            trade_type=TradeLog.TradeTypeChoices.BUY,
            order_type=TradeLog.OrderTypeChoices.LIMIT,
            price=price,
            quantity=quantity,
            commission=commission,
            stamp_duty=Decimal('0.00'),
            reason=TradeLog.ReasonChoices.ENTRY,
            status=TradeLog.StatusChoices.FILLED
        )
        
        self.service.last_buy_trade_id = trade_log.trade_id

    @transaction.atomic
    def sell_stock_by_market_price(self, position: Position, reason: str) -> None:
        if reason == TradeLog.ReasonChoices.STOP_LOSS:
            trigger_price = position.current_stop_loss
        elif reason == TradeLog.ReasonChoices.TAKE_PROFIT:
            trigger_price = position.current_take_profit
        else:
            trigger_price = self.get_opening_price(position.stock_code_id)

        sell_price = trigger_price * (Decimal('1.0') - self.service.SELL_SLIPPAGE_RATE)
        sell_price = sell_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        amount = sell_price * position.quantity
        commission = max(amount * self.service.COMMISSION_RATE, self.service.MIN_COMMISSION)
        stamp_duty = amount * self.service.STAMP_DUTY_RATE
        net_income = amount - commission - stamp_duty

        self.service.cash_balance += net_income
        logger.info(f"[回测] 卖出 {position.stock_code_id} {position.quantity}股 @{sell_price:.2f} (触发价: {trigger_price:.2f}), "
                    f"收入: {amount:.2f}, 佣金: {commission:.2f}, 印花税: {stamp_duty:.2f}, "
                    f"现金余额: {self.service.cash_balance:.2f}")

        position.status = Position.StatusChoices.CLOSED
        position.save()

        # 细节优化：使用更真实的卖出时间，例如下午2:57
        sell_time = time(14, 57, 0)
        sell_datetime = timezone.make_aware(
            timezone.datetime.combine(self.service.current_date, sell_time)
        )

        TradeLog.objects.create(
            position=position,
            stock_code_id=position.stock_code_id,
            trade_datetime=sell_datetime,
            trade_type=TradeLog.TradeTypeChoices.SELL,
            order_type=TradeLog.OrderTypeChoices.MARKET,
            price=sell_price,
            quantity=position.quantity,
            commission=commission,
            stamp_duty=stamp_duty,
            reason=reason,
            status=TradeLog.StatusChoices.FILLED
        )
