from django.urls import path
from . import views

urlpatterns = [
    # 选股管理
    path('selection/plans', views.selection_plans),
    path('selection/factors', views.selection_factors),
    path('selection/run', views.selection_run),
    # 通用-股票搜索
    path('stock/search', views.stock_search),
    # 日线管理
    path('daily/csi300', views.daily_csi300),
    path('daily/csi300/fetch', views.daily_csi300_fetch),
    path('daily/stock', views.daily_stock),
    path('daily/fetch', views.daily_fetch),
    path('daily/m_value', views.daily_m_value),
    # 因子管理
    path('factors/params', views.factors_params),
    path('factors/definitions', views.factors_definitions),
    # 系统管理
    path('system/schema', views.system_schema),
    path('system/backtest/results', views.system_backtest_results),
]


