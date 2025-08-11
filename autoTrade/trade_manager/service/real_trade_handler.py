# trade_manager/service/real_trade_handler.py

import logging
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, time, datetime, timedelta

import easytrader
import akshare as ak
from django.db import transaction
from django.utils import timezone

from .trade_handler import ITradeHandler
from common.models import Position, TradeLog, DailyQuotes
from trade_manager.service.decision_order_service import DecisionOrderService
from common.config_loader import config_loader # 使用统一的配置加载器

logger = logging.getLogger(__name__)

class ConnectionManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConnectionManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.user = None
            self.last_connected_date = None
            self.last_refreshed_time = None # 新增：上次刷新时间
            self.refresh_interval = timedelta(seconds=5) # 新增：刷新间隔
            self.initialized = True
            logger.info("ConnectionManager 已初始化。")

    def get_user(self):
        """获取或创建当天的 easytrader 连接，并按需刷新"""
        config = config_loader.get('easytrader')
        today = date.today()
        
        if not self.user or self.last_connected_date != today:
            logger.info("当天首次连接或连接已失效，正在重新建立 easytrader 连接...")
            try:
                self._connect(config)
                self.last_connected_date = today
                self.last_refreshed_time = datetime.now()
                logger.info("easytrader 连接成功。")
            except Exception as e:
                logger.error(f"连接 easytrader 失败: {e}", exc_info=True)
                self.user = None
                self.last_connected_date = None
                raise
        
        # 检查是否需要刷新
        if datetime.now() - self.last_refreshed_time > self.refresh_interval:
            logger.info("会话超过5分钟未刷新，执行 user.refresh()...")
            try:
                self.user.refresh()
                self.last_refreshed_time = datetime.now()
                logger.info("user.refresh() 执行成功。")
            except Exception as e:
                logger.error(f"执行 user.refresh() 失败: {e}，将尝试断开重连。")
                self.disconnect() # 刷新失败，可能连接已断，强制断开
                # 下次调用 get_user 时会自动重连
                raise # 抛出异常，让当前操作失败
        
        return self.user

    def _connect(self, config: dict):
        client_type = config.get('client_type', 'ht_client')
        user_config_path = config.get('user_config_path')
        
        if client_type == 'ht_client':
            self.user = easytrader.use('ht_client')
            self.user.prepare(user_config_path)
        else:
            raise NotImplementedError(f"不支持的客户端类型: {client_type}")

    def disconnect(self):
        if self.user:
            try:
                self.user.exit()
                logger.info("easytrader 连接已成功断开。")
            except Exception as e:
                logger.error(f"断开 easytrader 连接时出错: {e}", exc_info=True)
            finally:
                self.user = None
                self.last_connected_date = None
                self.last_refreshed_time = None

connection_manager = ConnectionManager()

class RealTradeHandler(ITradeHandler):
    COMMISSION_RATE = Decimal('0.00025')
    MIN_COMMISSION = Decimal('5')
    STAMP_DUTY_RATE = Decimal('0.001')

    def __init__(self):
        config = config_loader.get_config()
        self.is_simulation = (config.get('trading_mode') == 'real_simulation_observation')
        logger.info(f"RealTradeHandler 初始化。模式: {'实盘模拟观测' if self.is_simulation else '实盘交易'}")

    def _get_user(self):
        return connection_manager.get_user()

    def _api_buy(self, stock_code: str, price: Decimal, quantity: int):
        user = self._get_user()
        ak_code = stock_code.split('.')[-1]
        return user.buy(ak_code, price=float(price), amount=quantity)

    def _api_sell(self, stock_code: str, quantity: int):
        user = self._get_user()
        ak_code = stock_code.split('.')[-1]
        return user.sell(ak_code, amount=quantity)

    def _api_get_orders(self):
        user = self._get_user()
        return user.entrust

    def _api_get_balance(self):
        user = self._get_user()
        return user.balance

    def _api_get_realtime_quote(self, stock_code: str) -> dict:
        ak_code = stock_code.split('.')[-1]
        try:
            df = ak.stock_zh_a_spot_em(symbol=ak_code)
            if not df.empty:
                quote = df.iloc[0]
                return {
                    'open': Decimal(str(quote['今开'])),
                    'price': Decimal(str(quote['最新价'])),
                }
        except Exception as e:
            logger.warning(f"通过 akshare 获取 {stock_code} 实时行情失败: {e}")
        return {}

    def get_opening_price(self, stock_code: str) -> Decimal:
        quote = self._api_get_realtime_quote(stock_code)
        return quote.get('open', Decimal('0.00'))

    def get_realtime_price(self, stock_code: str) -> Decimal | None:
        quote = self._api_get_realtime_quote(stock_code)
        return quote.get('price')

    def get_available_balance(self) -> Decimal:
        if self.is_simulation:
            return Decimal('1000000.00')
        
        balance_info = self._api_get_balance()
        return Decimal(str(balance_info.get('可用金额', '0.00')))

    @transaction.atomic
    def place_buy_order(self, stock_code: str, price: Decimal, quantity: int):
        logger.info(f"准备下单买入: {stock_code}, 价格: {price}, 数量: {quantity}")
        
        entry_datetime = timezone.now()
        position = Position.objects.create(
            stock_code_id=stock_code, entry_datetime=entry_datetime,
            entry_price=price, quantity=quantity,
            current_stop_loss=Decimal('0.00'), current_take_profit=Decimal('0.00'),
            status=Position.StatusChoices.OPEN
        )
        trade_log = TradeLog.objects.create(
            position=position, stock_code_id=stock_code,
            trade_datetime=entry_datetime, trade_type=TradeLog.TradeTypeChoices.BUY,
            order_type=TradeLog.OrderTypeChoices.LIMIT, price=price,
            quantity=quantity, commission=Decimal('0.00'), stamp_duty=Decimal('0.00'),
            reason=TradeLog.ReasonChoices.ENTRY, status=TradeLog.StatusChoices.PENDING
        )

        if self.is_simulation:
            logger.info("[模拟模式] 跳过真实API调用，直接模拟成交。")
            amount = price * quantity
            commission = max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
            trade_log.status = TradeLog.StatusChoices.FILLED
            trade_log.commission = commission.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            trade_log.save()
            
            decision_service = DecisionOrderService(self, execution_date=date.today())
            decision_service.calculate_stop_profit_loss(trade_log.trade_id)
        else:
            try:
                order_result = self._api_buy(stock_code, price, quantity)
                logger.info(f"真实买入委托已提交: {order_result}")
                # 关键：保存委托编号
                if order_result and 'entrust_no' in order_result:
                    trade_log.external_order_id = str(order_result['entrust_no'])
                    trade_log.save()
            except Exception as e:
                logger.error(f"提交买入委托失败: {e}", exc_info=True)
                trade_log.status = TradeLog.StatusChoices.FAILED
                trade_log.save()
                position.status = Position.StatusChoices.CLOSED
                position.save()

    @transaction.atomic
    def sell_stock_by_market_price(self, position: Position, reason: str):
        logger.info(f"准备市价卖出: {position.stock_code_id}, 数量: {position.quantity}, 原因: {reason}")

        trade_log = TradeLog.objects.create(
            position=position, stock_code_id=position.stock_code_id,
            trade_datetime=timezone.now(), trade_type=TradeLog.TradeTypeChoices.SELL,
            order_type=TradeLog.OrderTypeChoices.MARKET, price=Decimal('0.00'),
            quantity=position.quantity, commission=Decimal('0.00'), stamp_duty=Decimal('0.00'),
            reason=reason, status=TradeLog.StatusChoices.PENDING
        )

        if self.is_simulation:
            logger.info("[模拟模式] 跳过真实API调用，直接模拟成交。")
            try:
                last_quote = DailyQuotes.objects.filter(stock_code_id=position.stock_code_id).latest('trade_date')
                sell_price = last_quote.close
            except DailyQuotes.DoesNotExist:
                sell_price = position.entry_price

            amount = sell_price * position.quantity
            commission = max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
            stamp_duty = amount * self.STAMP_DUTY_RATE

            trade_log.status = TradeLog.StatusChoices.FILLED
            trade_log.price = sell_price
            trade_log.commission = commission.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            trade_log.stamp_duty = stamp_duty.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            trade_log.save()

            position.status = Position.StatusChoices.CLOSED
            position.save()
        else:
            try:
                order_result = self._api_sell(position.stock_code_id, position.quantity)
                logger.info(f"真实卖出委托已提交: {order_result}")
                if order_result and 'entrust_no' in order_result:
                    trade_log.external_order_id = str(order_result['entrust_no'])
                    trade_log.save()
            except Exception as e:
                logger.error(f"提交卖出委托失败: {e}", exc_info=True)
                trade_log.status = TradeLog.StatusChoices.FAILED
                trade_log.save()
