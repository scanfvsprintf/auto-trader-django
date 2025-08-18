from django.db import models
from .stock_info import StockInfo

class DailyTradingPlan(models.Model):
    """
    3.1. 每日交易预案表 (tb_daily_trading_plan)
    说明: 存储 T-1 日终选股模块生成的“次日观察池”及相关交易预案。
    """
    class StatusChoices(models.TextChoices):
        PENDING = 'pending', '待执行'
        EXECUTED = 'executed', '已执行买入'
        CANCELLED = 'cancelled', '当日未满足条件作废'

    plan_date = models.DateField(
        help_text="预案执行日期 (T日)"
    )
    stock_code = models.ForeignKey(
        StockInfo, 
        on_delete=models.CASCADE, 
        db_column='stock_code',
        help_text="候选股票代码"
    )
    rank = models.IntegerField(
        help_text="综合得分排名 (1-10)"
    )
    final_score = models.DecimalField(
        max_digits=10, 
        decimal_places=4, 
        help_text="f(x)选股综合得分"
    )
    miop = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="最低可接受开盘价 (Minimum Acceptable Open Price)"
    )
    maop = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        help_text="最高可接受开盘价 (Maximum Acceptable Open Price)"
    )
    status = models.CharField(
        max_length=20, 
        choices=StatusChoices.choices, 
        default=StatusChoices.PENDING,
        help_text="预案状态。枚举: pending(待执行), executed(已执行买入), cancelled(当日未满足条件作废)"
    )
    strategy_dna = models.CharField(
        max_length=255,
        null=True,  # 允许为空，确保对旧数据和现有代码的兼容性
        blank=True, # 允许为空
        help_text="策略DNA贡献度, 格式: MT:0.70|BO:0.20|MR:0.05|QD:0.05"
    )
    def __str__(self):
        return f"Plan for {self.stock_code} on {self.plan_date} (Rank: {self.rank})"

    class Meta:
        db_table = 'tb_daily_trading_plan'
        # 使用 unique_together 实现复合主键的唯一性约束
        unique_together = (('plan_date', 'stock_code'),)
        verbose_name = '每日交易预案'
        verbose_name_plural = verbose_name
