from django.db import models
from django.utils import timezone

class FundInfo(models.Model):
    """
    基金基础信息表 (tb_fund_info)
    说明: 存储所有基金的基本信息，如代码、名称、类型等，作为其他数据表的关联基础。
    """
    class FundTypeChoices(models.TextChoices):
        OTHER = '000', '其他'
        ETF = '001', '场内ETF'
        # 后续可以在这里添加新的基金类型

    class StatusChoices(models.TextChoices):
        LISTING = 'listing', '上市'
        DELISTED = 'delisted', '退市'
        SUSPENDED = 'suspended', '停牌'

    fund_code = models.CharField(
        max_length=50, 
        primary_key=True, 
        help_text="基金代码, 格式如 'sh.510050'"
    )
    fund_name = models.CharField(
        max_length=50, 
        help_text="基金名称"
    )
    fund_type = models.CharField(
        max_length=3,
        choices=FundTypeChoices.choices,
        default=FundTypeChoices.OTHER,
        help_text="基金类型。枚举: 000(其他), 001(场内ETF)"
    )
    listing_date = models.DateField(
        help_text="上市日期"
    )
    status = models.CharField(
        max_length=20, 
        choices=StatusChoices.choices,
        help_text="基金状态。枚举: listing(上市), delisted(退市), suspended(停牌)"
    )
    created_at = models.DateTimeField(
        default=timezone.now, 
        editable=False,
        help_text="记录创建时间"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="记录更新时间"
    )

    def __str__(self):
        return f"{self.fund_name}({self.fund_code})"

    class Meta:
        db_table = 'tb_fund_info'
        verbose_name = '基金基础信息'
        verbose_name_plural = verbose_name
