from django.db import models
from .stock_info import StockInfo
from .factor_definitions import FactorDefinitions

class DailyFactorValues(models.Model):
    """
    2.2. 每日因子值表 (tb_daily_factor_values)
    说明: (核心设计) 存储每只股票在每个交易日计算出的所有因子原始值和标准化分值。
    """
    stock_code = models.ForeignKey(
        StockInfo, 
        on_delete=models.CASCADE, 
        db_column='stock_code',
        help_text="股票代码"
    )
    trade_date = models.DateField(
        help_text="交易日期"
    )
    factor_code = models.ForeignKey(
        FactorDefinitions, 
        on_delete=models.CASCADE, 
        db_column='factor_code',
        help_text="因子代码"
    )
    raw_value = models.DecimalField(
        max_digits=20, 
        decimal_places=10, 
        help_text="因子计算出的原始值"
    )
    norm_score = models.DecimalField(
        max_digits=10, 
        decimal_places=4, 
        null=True, 
        blank=True,
        help_text="经过norm()函数标准化后的分值 (-100到100)"
    )

    def __str__(self):
        return f"{self.stock_code} - {self.factor_code} on {self.trade_date}"

    class Meta:
        db_table = 'tb_daily_factor_values'
        # 使用 unique_together 实现复合主键的唯一性约束
        unique_together = (('stock_code', 'trade_date', 'factor_code'),)
        verbose_name = '每日因子值'
        verbose_name_plural = verbose_name
