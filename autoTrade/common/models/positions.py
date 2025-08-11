from django.db import models
from .stock_info import StockInfo

class Position(models.Model):
    """
    3.2. 持仓信息表 (tb_positions)
    说明: 存储当前所有持仓的详细信息，是盘中监控模块的核心数据依据。
    """
    class StatusChoices(models.TextChoices):
        OPEN = 'open', '持仓中'
        CLOSED = 'closed', '已平仓'

    position_id = models.BigAutoField(
        primary_key=True, 
        help_text="持仓唯一ID"
    )
    stock_code = models.ForeignKey(
        StockInfo, 
        on_delete=models.PROTECT, # 保护，防止意外删除关联股票信息
        db_column='stock_code',
        help_text="股票代码"
    )
    entry_datetime = models.DateTimeField(
        help_text="建仓成交时间"
    )
    entry_price = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="实际成交均价 (AEP)"
    )
    quantity = models.BigIntegerField(
        help_text="持仓数量 (股)"
    )
    current_stop_loss = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="当前止损价"
    )
    current_take_profit = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="当前止盈价"
    )
    status = models.CharField(
        max_length=20, 
        choices=StatusChoices.choices, 
        default=StatusChoices.OPEN,
        help_text="持仓状态。枚举: open(持仓中), closed(已平仓)"
    )

    def __str__(self):
        return f"Position {self.position_id}: {self.quantity} of {self.stock_code}"

    class Meta:
        db_table = 'tb_positions'
        verbose_name = '持仓信息'
        verbose_name_plural = verbose_name
