# trade_manager/service/monitor_exit_service.py

import logging
from datetime import date
from django.utils import timezone
from decimal import Decimal

# 导入项目内的模型和接口
from common.models import Position, TradeLog
from .trade_handler import ITradeHandler
from .position_monitor_logic import PositionMonitorLogic
persistent_logger = logging.getLogger(__name__)


class MonitorExitService:
    """
    3.5 - 盘中持仓监控与退出模块

    该服务负责在交易时段内，以固定频率轮询，监控所有非当日建仓的持仓。
    当持仓股票的实时价格触及预设的止盈或止损线时，调用交易处理器执行卖出操作。
    """
    MODULE_NAME = '盘中持仓监控与退出'

    def __init__(self, handler: ITradeHandler,execution_date: date = None):
        """
        初始化监控服务。

        :param handler: 一个实现了 ITradeHandler 接口的实例，用于与交易环境交互。
        """
        if not isinstance(handler, ITradeHandler):
            raise TypeError("传入的 handler 必须是 ITradeHandler 的一个实例。")
        
        self.handler = handler
        self.execution_date = execution_date if execution_date else timezone.now().date()
        # 使用特定的logger进行高频、非持久化的日志记录
        self.logger = persistent_logger

    def monitor_and_exit_positions(self):
        """
        执行一次完整的持仓监控与退出检查。
        此函数应由一个定时调度器在交易时段内（09:30:01 - 14:57:00）
        以设定的频率反复调用。
        """
        self.logger.debug(f"[{self.MODULE_NAME}] 任务开始...")

        # 1. 从持仓信息表读取出entry_datetime建仓成交时间不为今天的持仓信息
        today = timezone.now().date()
        positions_to_monitor = Position.objects.filter(
            status=Position.StatusChoices.OPEN
        ).exclude(
            entry_datetime__date=self.execution_date
        )

        if not positions_to_monitor.exists():
            self.logger.debug("当前无需要监控的隔夜持仓。")
            return

        # 2. 循环调用处理器判断是否达到了止盈止损状态
        for position in positions_to_monitor:
            try:
                # 获取实时价格
                current_price = self.handler.get_realtime_price(position.stock_code)

                if current_price is None or current_price <= 0:
                    self.logger.debug(f"无法获取 {position.stock_code} 的有效实时价格，跳过本次检查。")
                    continue
                
                self.logger.debug(
                    f"监控: {position.stock_code}, "
                    f"当前价: {current_price}, "
                    f"止损价: {position.current_stop_loss}, "
                    f"止盈价: {position.current_take_profit}"
                )

                # 调用中央决策逻辑
                decision = PositionMonitorLogic.check_and_decide(position, current_price, self.params)
                if decision['action'] == 'SELL':
                    msg = f"触发卖出! 股票: {position.stock_code_id}, 价格: {current_price:.2f}, 机制原因: {decision['reason']}"
                    persistent_logger.info(msg)
                    # 注意：实盘卖出时，成交价未知，所以reason是基于触发机制的
                    self.handler.sell_stock_by_market_price(position, decision['reason'])
                
                elif decision['action'] == 'UPDATE':
                    updates = decision['updates']
                    for field, value in updates.items():
                        setattr(position, field, value)
                    position.save(update_fields=list(updates.keys()))
                    persistent_logger.info(f"风控价格更新! 股票: {position.stock_code_id}, 更新内容: {updates}")

            except Exception as e:
                # 根据要求，卖出失败等异常只在控制台打印错误日志，等待下一次循环
                self.logger.error(
                    f"处理持仓 {position.position_id} ({position.stock_code}) 时发生错误: {e}",
                    exc_info=False # 在高频场景下，可以关闭traceback以保持日志简洁
                )
                continue
        
        self.logger.debug(f"[{self.MODULE_NAME}] 任务结束。")

