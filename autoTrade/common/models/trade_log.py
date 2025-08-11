from django.db import models
from .stock_info import StockInfo

class TradeLog(models.Model):
    """
    3.3. 交易记录表 (tb_trade_log)
    说明: 记录每一次买入和卖出的详细信息，用于成本核算、业绩分析和问题排查。
    """
    class TradeTypeChoices(models.TextChoices):
        BUY = 'buy', '买入'
        SELL = 'sell', '卖出'

    class OrderTypeChoices(models.TextChoices):
        LIMIT = 'limit', '限价'
        MARKET = 'market', '市价'

    class ReasonChoices(models.TextChoices):
        ENTRY = 'entry', '策略入场'
        TAKE_PROFIT = 'take_profit', '止盈'
        STOP_LOSS = 'stop_loss', '止损'
        MANUAL = 'manual', '人工干预'

    class StatusChoices(models.TextChoices):
        FILLED = 'filled', '已成交'
        FAILED = 'failed', '失败'
        CANCELLED = 'cancelled', '已撤销'
        PENDING = 'pending','待执行'

    trade_id = models.BigAutoField(
        primary_key=True, 
        help_text="交易唯一ID"
    )
    # 注意：这里使用字符串 'positions.Position' 来避免循环导入问题
    # related_name='trade_logs' 允许从 Position 对象反向访问其所有交易记录
    position = models.ForeignKey(
        'Position', 
        on_delete=models.CASCADE, # 如果持仓被删除，关联的交易记录也应删除
        related_name='trade_logs',
        help_text="关联的持仓ID (买入时生成, 卖出时引用)"
    )
    stock_code = models.ForeignKey(
        StockInfo, 
        on_delete=models.PROTECT,
        db_column='stock_code',
        help_text="股票代码"
    )
    trade_datetime = models.DateTimeField(
        help_text="交易成交时间"
    )
    trade_type = models.CharField(
        max_length=10, 
        choices=TradeTypeChoices.choices,
        help_text="交易类型。枚举: buy(买入), sell(卖出)"
    )
    order_type = models.CharField(
        max_length=10, 
        choices=OrderTypeChoices.choices,
        help_text="订单类型。枚举: limit(限价), market(市价)"
    )
    price = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="成交均价"
    )
    quantity = models.BigIntegerField(
        help_text="成交数量"
    )
    commission = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="佣金"
    )
    stamp_duty = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        default=0,
        help_text="印花税 (仅卖出时有)"
    )
    reason = models.CharField(
        max_length=50, 
        choices=ReasonChoices.choices, 
        blank=True, 
        null=True,
        help_text="交易原因。枚举: entry(策略入场), take_profit(止盈), stop_loss(止损), manual(人工干预)"
    )
    status = models.CharField(
        max_length=20, 
        choices=StatusChoices.choices,
        help_text="订单状态。枚举: filled(已成交), failed(失败), cancelled(已撤销),pending(待执行)"
    )

    external_order_id = models.CharField(
        max_length=50, 
        null=True, 
        blank=True, 
        db_index=True,
        help_text="外部交易系统的订单ID，如券商的委托编号"
    )

    def __str__(self):
        return f"Trade {self.trade_id}: {self.trade_type.upper()} {self.quantity} of {self.stock_code}"

    class Meta:
        db_table = 'tb_trade_log'
        verbose_name = '交易记录'
        verbose_name_plural = verbose_name
