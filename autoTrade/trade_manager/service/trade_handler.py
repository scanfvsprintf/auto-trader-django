# trade_manager/service/trade_handler.py

from abc import ABC, abstractmethod
from decimal import Decimal
from django.db import transaction
from django.utils import timezone
# 为了类型提示，我们可以从 common.models 导入 Position 和 TradeLog
# 注意：为了避免循环导入，通常在实现类中进行实际导入，这里仅为类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common.models import Position, TradeLog



class ITradeHandler(ABC):
    """
    交易处理器抽象基类 (Abstract Base Class)。

    该接口定义了开盘决策与下单模块所需的所有外部交互行为。
    通过依赖此抽象接口而非具体实现，`DecisionOrderService` 可以与不同的
    交易环境（如真实交易接口、回测引擎）解耦。

    - 对于真实交易，实现类将通过API与券商服务器交互。
    - 对于回测，实现类将模拟这些交互，例如从历史数据中读取开盘价、
      模拟订单成交、并管理一个虚拟账户的余额。
    """

    @abstractmethod
    def get_opening_price(self, stock_code: str) -> Decimal:
        """
        获取指定股票在执行日的实际开盘价。

        :param stock_code: 股票代码，格式与 tb_stock_info 表一致 (如 'sh.600000')。
        :return: 当日的开盘价。如果无法获取（例如停牌），应引发异常或返回一个可识别的错误值（如Decimal('0.00')）。
        """
        pass

    @abstractmethod
    def place_buy_order(self, stock_code: str, price: Decimal, quantity: int) -> None:
        """
        提交一个买入订单。

        此方法的实现者负责处理与交易系统的所有交互。根据需求，此方法
        在执行时，需要完成以下数据库操作：
        1. 在 `tb_positions` 表中插入一条新的持仓记录，其中所有非空字段
           （如 current_stop_loss, current_take_profit）可使用哨兵值（如-1）填充，
           等待后续的止盈止损计算任务来更新。
        2. 在 `tb_trade_log` 表中插入一条对应的交易记录，初始状态应为
           'pending'。

        :param stock_code: 股票代码。
        :param price: 预期的买入限价。
        :param quantity: 计划买入的股数（必须是100的整数倍）。
        :return: 无返回值。
        """
        pass

    @abstractmethod
    def get_available_balance(self) -> Decimal:
        """
        查询当前账户的可用资金余额。

        :return: 可用于交易的现金余额。
        """
        pass

    
    @abstractmethod
    def get_realtime_price(self, stock_code: str) -> Decimal | None:
        """
        获取一只股票的实时价格。
 
        :param stock_code: 股票代码，格式如 'sh.600000'。
        :return: 该股票此时此刻的市场价 (Decimal类型)。如果获取失败（如网络问题、股票停牌），
                 应返回 None，以便调用方进行错误处理。
        """
        pass
 
    @abstractmethod
    def sell_stock_by_market_price(self, position: 'Position', reason: str) -> None:
        """
        以市价单全量卖出指定的持仓。
 
        此方法的具体实现需要完成一个原子性的操作流程：
        1. 调用交易API，以市价单卖出 `position.quantity` 数量的 `position.stock_code`。
        2. **在API调用成功返回成交回报后**，执行以下数据库操作：
           a. **更新持仓表 (tb_positions)**: 将传入的 `position` 对象的状态更新为 'closed'。
              `position.status = Position.StatusChoices.CLOSED`
              `position.save()`
           b. **插入交易记录 (tb_trade_log)**: 创建一条新的卖出记录。
              - `position`: 关联到此持仓。
              - `stock_code`: 股票代码。
              - `trade_datetime`: 交易的实际成交时间。
              - `trade_type`: 'sell'。
              - `order_type`: 'market'。
              - `quantity`: 实际成交数量。
              - `price`: 实际的成交均价。从成交回报中获取。如果无法立即获取，
                                            则使用 -1 作为占位符，等待后续任务回补。
              - `commission`, `stamp_duty`: 从成交回报中获取。如果无法立即获取，
                                            则使用 -1 作为占位符，等待后续任务回补。
              - `reason`: 使用传入的 `reason` 参数 ('take_profit' 或 'stop_loss')。
              - `status`: 'filled' (已成交)。
        3. 整个数据库更新过程应该被包裹在一个事务中 (`transaction.atomic`)，确保数据一致性。
 
        :param position: 要卖出的持仓对象 (common.models.positions.Position)。
                         该对象包含了持仓ID、股票代码、持仓数量等所有必要信息。
        :param reason: 卖出原因的字符串，如 'take_profit' 或 'stop_loss'。
                       这个值将用于填充交易记录表的 `reason` 字段。
        :return: 无返回值。如果执行失败（如API调用失败、股票跌停无法卖出），
                 应在方法内部处理异常（如记录日志），并向上层调用者（MonitorExitService）
                 抛出异常或通过其他方式通知失败，以便上层决定是否重试。
        """
        pass