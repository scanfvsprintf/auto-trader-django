from django.db import models

class FactorDefinitions(models.Model):
    """
    2.1. 因子定义表 (tb_factor_definitions)
    说明: (核心设计) 用于定义所有策略中使用的因子，实现因子的可插拔。新增因子只需在此表增加一条记录。
    """
    class DirectionChoices(models.TextChoices):
        POSITIVE = 'positive', '正向, 值越大越好'
        NEGATIVE = 'negative', '负向, 值越小越好'

    factor_code = models.CharField(
        max_length=50, 
        primary_key=True, 
        help_text="因子唯一英文代码, 如 'MA20_SLOPE'"
    )
    factor_name = models.CharField(
        max_length=100, 
        help_text="因子中文名称, 如 '20日均线斜率'"
    )
    description = models.TextField(
        blank=True, 
        null=True, 
        help_text="详细描述因子的计算逻辑和业务含义"
    )
    direction = models.CharField(
        max_length=10, 
        choices=DirectionChoices.choices,
        help_text="因子方向性。枚举: positive(正向, 值越大越好), negative(负向, 值越小越好)"
    )
    is_active = models.BooleanField(
        default=True, 
        help_text="是否启用该因子"
    )

    def __str__(self):
        return f"{self.factor_name} ({self.factor_code})"

    class Meta:
        db_table = 'tb_factor_definitions'
        verbose_name = '因子定义'
        verbose_name_plural = verbose_name
