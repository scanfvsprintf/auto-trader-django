from django.db import models

class StrategyParameters(models.Model):
    """
    2.3. 策略参数表 (tb_strategy_parameters)
    说明: 存储所有策略中可优化的参数，如权重、系数等，方便回测与优化模块进行读取和修改。
    """
    param_name = models.CharField(
        max_length=50, 
        primary_key=True, 
        help_text="参数唯一英文名, 如 'w_trend', 'k_h1'"
    )
    param_value = models.DecimalField(
        max_digits=20, 
        decimal_places=10, 
        help_text="参数的数值"
    )
    group_name = models.CharField(
        max_length=50, 
        blank=True, 
        null=True, 
        help_text="参数所属分组, 如 'WEIGHTS', 'STOP_LOSS'"
    )
    description = models.TextField(
        blank=True, 
        null=True, 
        help_text="参数的详细说明"
    )

    def __str__(self):
        return f"{self.param_name} = {self.param_value}"

    class Meta:
        db_table = 'tb_strategy_parameters'
        verbose_name = '策略参数'
        verbose_name_plural = verbose_name
