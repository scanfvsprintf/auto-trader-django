from django.db import models
from .fund_info import FundInfo

class FundDailyQuotes(models.Model):
    """
    基金日线行情表 (tb_fund_daily_quotes)
    说明: 存储从数据源获取的最原始的基金日线行情数据，是所有计算的基石。
    """
    fund_code = models.ForeignKey(
        FundInfo, 
        on_delete=models.CASCADE, 
        db_column='fund_code',
        help_text="基金代码"
    )
    trade_date = models.DateField(
        help_text="交易日期"
    )
    open = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="不复权开盘价"
    )
    high = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="不复权最高价"
    )
    low = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="不复权最低价"
    )
    close = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="不复权收盘价"
    )
    volume = models.BigIntegerField(
        help_text="成交量 (股)"
    )
    turnover = models.DecimalField(
        max_digits=20, 
        decimal_places=2, 
        help_text="成交额 (元)"
    )
    adjust_factor = models.DecimalField(
        max_digits=20, 
        decimal_places=10, 
        help_text="截至当日的后复权因子"
    )
    # 计算列 hfq_close
    hfq_close = models.DecimalField(
        max_digits=20, 
        decimal_places=10, 
        editable=False,
        help_text="后复权收盘价，公式: close * adjust_factor"
    )

    def save(self, *args, **kwargs):
        # 在保存模型前计算 hfq_close 的值
        self.hfq_close = self.close * self.adjust_factor
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.fund_code} on {self.trade_date}"

    class Meta:
        db_table = 'tb_fund_daily_quotes'
        # 使用 unique_together 实现复合主键的唯一性约束
        unique_together = (('fund_code', 'trade_date'),)
        verbose_name = '基金日线行情'
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date'], name='funddailyquotes_tradedate_idx'),
        ]
