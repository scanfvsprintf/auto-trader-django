# ==============================================================================
# 描述: 定义回测日志相关的新模型。
# ==============================================================================
from django.db import models

class BacktestDailyLog(models.Model):
    """
    回测每日日志表
    
    说明: 记录每一次回测运行中，每个交易日结束后的资金、持仓和市场状态。
    这张表用于生成资金曲线图和关键的回撤指标。
    """
    backtest_run_id = models.CharField(
        max_length=50,
        db_index=True,
        help_text="回测运行的唯一标识符, 如 'backtest_20231027_153000'"
    )
    trade_date = models.DateField(
        help_text="交易日期"
    )
    total_assets = models.DecimalField(
        max_digits=20,
        decimal_places=4,
        help_text="当日日终总资产 (现金 + 持仓市值)"
    )
    cash = models.DecimalField(
        max_digits=20,
        decimal_places=4,
        help_text="当日日终现金余额"
    )
    holdings_value = models.DecimalField(
        max_digits=20,
        decimal_places=4,
        help_text="当日日终持仓市值"
    )
    market_m_value = models.DecimalField(
        max_digits=18,
        decimal_places=10,
        null=True,
        blank=True,
        help_text="当日的市场状态M(t)值"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="记录创建时间"
    )

    class Meta:
        db_table = 'tb_trade_manager_backtest_daily_log'
        verbose_name = '回测每日日志'
        verbose_name_plural = verbose_name
        ordering = ['backtest_run_id', 'trade_date']
        # 为常用查询添加索引
        indexes = [
            models.Index(fields=['backtest_run_id', 'trade_date']),
        ]

class BacktestOperationLog(models.Model):
    """
    回测操作记录表
    
    说明: 增量记录回测过程中的每一次买入和卖出操作。
    这张表用于计算胜率、收益贡献等交易层面的指标。
    """
    class Direction(models.TextChoices):
        BUY = 'BUY', '买入'
        SELL = 'SELL', '卖出'

    class ExitReason(models.TextChoices):
        TAKE_PROFIT = 'TAKE_PROFIT', '止盈'
        STOP_LOSS = 'STOP_LOSS', '止损'

    backtest_run_id = models.CharField(
        max_length=50,
        db_index=True,
        help_text="回测运行的唯一标识符"
    )
    position_id_ref = models.BigIntegerField(
        db_index=True,
        help_text="关联的持仓ID (tb_positions.position_id)，用于反查"
    )
    stock_code = models.CharField(
        max_length=50,
        help_text="股票代码, 如 'sh.600000'"
    )
    stock_name = models.CharField(
        max_length=50,
        help_text="股票名称"
    )
    trade_date = models.DateField(
        help_text="交易发生的日期"
    )
    direction = models.CharField(
        max_length=10,
        choices=Direction.choices,
        help_text="买卖方向"
    )
    exit_reason = models.CharField(
        max_length=20,
        choices=ExitReason.choices,
        null=True,
        blank=True,
        help_text="止盈/止损方向 (仅卖出时有效)"
    )
    profit_rate = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="该笔交易设置的止盈率"
    )
    loss_rate = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="该笔交易设置的止损率"
    )
    buy_date_m_value = models.DecimalField(
        max_digits=18,
        decimal_places=10,
        null=True,
        blank=True,
        help_text="买入决策所依据的T-1日市场M(t)值"
    )
    factor_scores = models.TextField(
        help_text="买入时各因子得分, 格式: factor1:score1|factor2:score2"
    )
    price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        help_text="成交价格"
    )
    quantity = models.BigIntegerField(
        help_text="成交数量"
    )
    amount = models.DecimalField(
        max_digits=20,
        decimal_places=4,
        help_text="成交总金额"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="记录创建时间"
    )

    class Meta:
        db_table = 'tb_trade_manager_backtest_operation_log'
        verbose_name = '回测操作记录'
        verbose_name_plural = verbose_name
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['backtest_run_id', 'stock_code']),
            models.Index(fields=['backtest_run_id', 'position_id_ref']),
        ]

class MDistributionBacktestLog(models.Model):
    """
    M值胜率分布回测的专用日志表。
    
    说明: 每一条记录代表一次完整的模拟交易（从预案生成到最终平仓）。
    这张表是M值胜率分布回测报告的唯一数据源。
    """
    class ExitReason(models.TextChoices):
        TAKE_PROFIT = 'TAKE_PROFIT', '止盈'
        STOP_LOSS = 'STOP_LOSS', '止损'
        END_OF_PERIOD = 'END_OF_PERIOD', '达到最大持有期'
    backtest_run_id = models.CharField(
        max_length=50,
        db_index=True,
        help_text="回测运行的唯一标识符"
    )
    plan_date = models.DateField(
        help_text="预案生成日期 (T-1日)"
    )
    stock_code = models.CharField(
        max_length=50,
        help_text="股票代码"
    )
    stock_name = models.CharField(
        max_length=50,
        help_text="股票名称"
    )
    m_value_at_plan = models.DecimalField(
        max_digits=18,
        decimal_places=10,
        help_text="预案生成日的市场M(t)值"
    )
    strategy_dna = models.CharField(
        max_length=255,
        help_text="策略DNA贡献度, 格式: MT:0.70|BO:0.20|MR:0.05|QD:0.05"
    )
    entry_date = models.DateField(
        help_text="模拟入场日期 (T日)"
    )
    entry_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        help_text="模拟入场价格 (T日开盘价)"
    )
    exit_date = models.DateField(
        help_text="模拟出场日期"
    )
    exit_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        help_text="模拟出场价格"
    )
    exit_reason = models.CharField(
        max_length=20,
        choices=ExitReason.choices,
        help_text="平仓原因"
    )
    holding_period = models.IntegerField(
        help_text="持有天数（交易日）"
    )
    # 预设的止盈止损率，用于计算期望收益
    preset_take_profit_rate = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        help_text="根据策略计算出的预设止盈率"
    )
    preset_stop_loss_rate = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        help_text="根据策略计算出的预设止损率"
    )
    # 实际收益率
    actual_return_rate = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        help_text="该笔交易的实际收益率 ( (exit_price / entry_price) - 1 )"
    )
    created_at = models.DateTimeField(
        auto_now_add=True
    )
    class Meta:
        db_table = 'tb_m_distribution_backtest_log'
        verbose_name = 'M值分布回测日志'
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['backtest_run_id', 'plan_date']),
        ]