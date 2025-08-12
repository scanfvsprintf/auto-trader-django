# common/admin.py

from django.contrib import admin
from .models import (
    StockInfo, DailyQuotes, FactorDefinitions, DailyFactorValues,
    StrategyParameters, DailyTradingPlan, Position, TradeLog,
    SystemLog, CorporateAction
)

# -----------------------------------------------------------------------------
# 1. 基础数据管理 (股票信息、行情、股权事件)
# -----------------------------------------------------------------------------

@admin.register(StockInfo)
class StockInfoAdmin(admin.ModelAdmin):
    """股票基础信息管理"""
    list_display = ('stock_code', 'stock_name', 'listing_date', 'status', 'updated_at')
    search_fields = ('stock_code', 'stock_name')
    list_filter = ('status', 'listing_date')
    ordering = ('stock_code',)
    readonly_fields = ('created_at', 'updated_at')

@admin.register(DailyQuotes)
class DailyQuotesAdmin(admin.ModelAdmin):
    """日线行情管理"""
    list_display = ('trade_date', 'stock_code', 'open', 'close', 'volume', 'turnover', 'hfq_close')
    search_fields = ('stock_code__stock_code', 'stock_code__stock_name')
    list_filter = ('trade_date',)
    ordering = ('-trade_date', 'stock_code')
    # 关键性能优化：对于有成千上万条记录的外键，使用 raw_id_fields 替代下拉框
    raw_id_fields = ('stock_code',)
    readonly_fields = ('hfq_close',)
    list_per_page = 25 # 设置每页显示条数

class CorporateActionAdmin(admin.ModelAdmin):
    """股权事件管理"""
    list_display = ('ex_dividend_date', 'stock_code', 'event_type', 'dividend_per_share', 'shares_before', 'shares_after', 'rights_issue_price')
    # 修正：直接搜索本表的 stock_code 字段即可
    search_fields = ('stock_code',) 
    list_filter = ('event_type', 'ex_dividend_date')
    ordering = ('-ex_dividend_date', 'stock_code')
    # raw_id_fields = ('stock_code',)

# -----------------------------------------------------------------------------
# 2. 策略与因子定义管理
# -----------------------------------------------------------------------------

@admin.register(FactorDefinitions)
class FactorDefinitionsAdmin(admin.ModelAdmin):
    """因子定义管理"""
    list_display = ('factor_code', 'factor_name', 'direction', 'is_active', 'description')
    search_fields = ('factor_code', 'factor_name')
    list_filter = ('direction', 'is_active')
    ordering = ('factor_code',)

@admin.register(StrategyParameters)
class StrategyParametersAdmin(admin.ModelAdmin):
    """策略参数管理"""
    list_display = ('param_name', 'param_value', 'group_name', 'description')
    search_fields = ('param_name', 'group_name')
    list_filter = ('group_name',)
    ordering = ('group_name', 'param_name')
    # 核心功能：允许在列表页直接编辑参数值，非常方便调参
    list_editable = ('param_value',)

@admin.register(DailyFactorValues)
class DailyFactorValuesAdmin(admin.ModelAdmin):
    """每日因子值管理"""
    list_display = ('trade_date', 'stock_code', 'factor_code', 'raw_value', 'norm_score')
    search_fields = ('stock_code__stock_code', 'factor_code__factor_code')
    list_filter = ('trade_date', 'factor_code')
    ordering = ('-trade_date', 'stock_code')
    # 关键性能优化
    raw_id_fields = ('stock_code', 'factor_code')
    list_per_page = 25

# -----------------------------------------------------------------------------
# 3. 交易流程管理 (预案、持仓、记录)
# -----------------------------------------------------------------------------

@admin.register(DailyTradingPlan)
class DailyTradingPlanAdmin(admin.ModelAdmin):
    """每日交易预案管理"""
    list_display = ('plan_date', 'stock_code', 'rank', 'final_score', 'miop', 'maop', 'status')
    search_fields = ('stock_code__stock_code',)
    list_filter = ('plan_date', 'status')
    ordering = ('-plan_date', 'rank')
    raw_id_fields = ('stock_code',)
    list_per_page = 20

@admin.register(Position)
class PositionAdmin(admin.ModelAdmin):
    """持仓信息管理"""
    list_display = ('position_id', 'stock_code', 'entry_datetime', 'entry_price', 'quantity', 'current_stop_loss', 'current_take_profit', 'status')
    search_fields = ('stock_code__stock_code',)
    list_filter = ('status', 'entry_datetime')
    ordering = ('-entry_datetime',)
    raw_id_fields = ('stock_code',)

@admin.register(TradeLog)
class TradeLogAdmin(admin.ModelAdmin):
    """交易记录管理"""
    list_display = ('trade_id', 'position', 'stock_code', 'trade_datetime', 'trade_type', 'price', 'quantity', 'reason', 'status')
    search_fields = ('stock_code__stock_code', 'position__position_id')
    list_filter = ('trade_type', 'status', 'reason', 'trade_datetime')
    ordering = ('-trade_datetime',)
    raw_id_fields = ('position', 'stock_code')
    list_per_page = 25

# -----------------------------------------------------------------------------
# 4. 系统与日志管理
# -----------------------------------------------------------------------------

@admin.register(SystemLog)
class SystemLogAdmin(admin.ModelAdmin):
    """系统日志管理"""
    list_display = ('log_time', 'log_level', 'module_name', 'message_summary')
    list_filter = ('log_level', 'module_name', 'log_time')
    search_fields = ('message', 'module_name')
    ordering = ('-log_time',)
    # 日志应该是不可变的，所以设为只读
    readonly_fields = ('log_time', 'log_level', 'module_name', 'message')
    list_per_page = 30

    def message_summary(self, obj):
        """在列表页显示截断的日志信息"""
        return (obj.message[:80] + '...') if len(obj.message) > 80 else obj.message
    message_summary.short_description = '日志摘要'

    def has_add_permission(self, request):
        # 禁止在Admin后台手动添加日志
        return False

    def has_change_permission(self, request, obj=None):
        # 禁止在Admin后台修改日志
        return False
