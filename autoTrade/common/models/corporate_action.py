from django.db import models

class CorporateAction(models.Model):
    """
    股权事件表 (tb_corporate_actions)
    
    说明: 存储所有影响股价非交易性变动的股权事件，是盘前校准模块和回测引擎的核心数据源。
    
    例子:
    10送5：event_type='bonus', shares_before=10, shares_after=15
    10转3：event_type='transfer', shares_before=10, shares_after=13
    10配3，配股价8元：event_type='rights', shares_before=10, shares_after=13, rights_issue_price=8
    1拆2：event_type='split', shares_before=1, shares_after=2 (理解为在1股基础上增加1股)
    10并1：event_type='split', shares_before=10, shares_after=1 (理解为在10股基础上减少9股)
    派1元：event_type='dividend', dividend_per_share=0.1
    """

    # 使用 Django 推荐的 TextChoices 来定义事件类型的枚举
    class EventType(models.TextChoices):
        DIVIDEND = 'dividend', '分红'
        BONUS = 'bonus', '送股'
        TRANSFER = 'transfer', '转股'
        RIGHTS = 'rights', '配股'
        SPLIT = 'split', '拆股/并股'

    # 字段定义
    event_id = models.BigAutoField(
        primary_key=True,
        help_text="事件唯一ID"
    )
    stock_code = models.CharField(
        max_length=50,
        null=False,
        blank=False,
        help_text="股票代码, 格式如 'sh.600000'"
    )
    ex_dividend_date = models.DateField(
        null=False,
        db_index=True,
        help_text="除权除息日 (策略判断的基准日期)，对于配股来说，实际为股权登记日而非除权日"
    )
    record_date = models.DateField(
        null=True,
        blank=True,
        help_text="股权登记日"
    )
    notice_date = models.DateField(
        null=True,
        blank=True,
        help_text="公告日期"
    )
    event_type = models.CharField(
        max_length=20,
        choices=EventType.choices,
        null=False,
        blank=False,
        help_text="事件类型。枚举: dividend(分红), bonus(送股), transfer(转股),rights(配股), split(拆股/并股)"
    )
    dividend_per_share = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="每股派息(税前, 元，分红专用)"
    )
    shares_before = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="基准股数 (如“10送5”，此值为10，送股/转股/拆股/并股专用)"
    )
    shares_after = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="变动股数 (如“10送5”，此值为15，送股/转股/拆股/并股专用)"
    )
    rights_issue_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="配股价格，配股专用"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        null=False,
        help_text="记录创建时间"
    )

    class Meta:
        # 显式指定数据库中的表名
        db_table = 'tb_corporate_actions'
        # 在 Django Admin 中显示的名称
        verbose_name = '股权事件'
        verbose_name_plural = '股权事件'
        # 默认排序规则
        ordering = ['-ex_dividend_date', 'stock_code']

    def __str__(self):
        # 提供一个易于阅读的对象表示形式
        return f"{self.stock_code} on {self.ex_dividend_date}: {self.get_event_type_display()}"

