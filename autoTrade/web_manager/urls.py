from django.urls import path
from . import views

urlpatterns = [
    # 选股管理
    path('selection/plans', views.selection_plans),
    path('selection/factors', views.selection_factors),
    path('selection/run', views.selection_run),
    path('selection/run_range', views.selection_run_range),
    # 通用-股票搜索
    path('stock/search', views.stock_search),
    # 通用-ETF搜索
    path('etf/search', views.etf_search),
    # 日线管理
    path('daily/csi300', views.daily_csi300),
    path('daily/csi300/fetch', views.daily_csi300_fetch),
    path('daily/stock', views.daily_stock),
    path('daily/etf', views.daily_etf),
    path('daily/fetch', views.daily_fetch),
    path('daily/m_value', views.daily_m_value),
    # 因子管理
    path('factors/params', views.factors_params),
    path('factors/definitions', views.factors_definitions),
    # 系统管理
    path('system/schema', views.system_schema),
    path('system/backtest/results', views.system_backtest_results),
    # AI配置管理
    path('ai/source/config', views.ai_source_config),
    path('ai/model/config', views.ai_model_config),
    path('ai/generate/text', views.ai_generate_text),
    path('ai/test/connection', views.ai_test_connection),
    path('ai/available/models', views.ai_available_models),
    # AI评测功能
    path('ai/evaluate/csi300', views.ai_evaluate_csi300),
    path('ai/evaluate/stock', views.ai_evaluate_stock),
]


