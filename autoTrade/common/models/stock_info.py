from django.db import models
from django.utils import timezone

class StockInfo(models.Model):
    """
    1.1. 股票基础信息表 (tb_stock_info)
    说明: 存储所有A股股票的基本信息，如代码、名称、上市日期等，作为其他数据表的关联基础。
    """
    class StatusChoices(models.TextChoices):
        LISTING = 'listing', '上市'
        DELISTED = 'delisted', '退市'
        SUSPENDED = 'suspended', '停牌'

    stock_code = models.CharField(
        max_length=10, 
        primary_key=True, 
        help_text="股票代码, 格式如 'sh.600000'"
    )
    stock_name = models.CharField(
        max_length=50, 
        help_text="股票名称"
    )
    listing_date = models.DateField(
        help_text="上市日期, 用于剔除次新股"
    )
    status = models.CharField(
        max_length=20, 
        choices=StatusChoices.choices,
        help_text="股票状态。枚举: listing(上市), delisted(退市), suspended(停牌)"
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
        return f"{self.stock_name}({self.stock_code})"

    class Meta:
        db_table = 'tb_stock_info'
        verbose_name = '股票基础信息'
        verbose_name_plural = verbose_name
