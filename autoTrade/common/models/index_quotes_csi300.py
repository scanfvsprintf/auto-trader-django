# common/models/index_quotes_csi300.py
from django.db import models
from decimal import Decimal

class IndexQuotesCsi300(models.Model):
    """
    沪深300指数日线行情表

    说明: 存储沪深300指数的日线行情数据，作为市场状态函数 M(t) 的计算基石。
    数据通过 akshare 的 index_zh_a_hist 接口获取。
    """
    trade_date = models.DateField(
        primary_key=True,
        help_text="交易日期"
    )
    open = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="开盘价"
    )
    close = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="收盘价"
    )
    high = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="最高价"
    )
    low = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="最低价"
    )
    volume = models.BigIntegerField(
        help_text="成交量 (股)"
    )
    amount = models.DecimalField(
        max_digits=20, decimal_places=2, help_text="成交额 (元)"
    )
    amplitude = models.DecimalField(
        max_digits=8, decimal_places=4, help_text="振幅 (%)"
    )
    pct_change = models.DecimalField(
        max_digits=8, decimal_places=4, help_text="涨跌幅 (%)"
    )
    change_amount = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="涨跌额"
    )
    turnover_rate = models.DecimalField(
        max_digits=8, decimal_places=4, null=True, blank=True, help_text="换手率 (%)"
    )

    class Meta:
        db_table = 'tb_index_quotes_csi300'
        verbose_name = '沪深300指数行情'
        verbose_name_plural = verbose_name
        ordering = ['-trade_date']

    def __str__(self):
        return f"CSI300 on {self.trade_date}: {self.close}"
