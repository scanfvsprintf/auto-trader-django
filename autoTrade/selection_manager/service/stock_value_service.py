# selection_manager/service/stock_value_service.py
# [修改后]
# 描述: 提供个股模型评分的服务。
#       - 使用新的 StockFactorCalculator 进行因子计算。
#       - 增加了在实时模式下批量加载数据并构建面板的能力。
#       - 能够处理回测时传入的预加载面板数据。

import logging
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from django.conf import settings

from common.models import DailyQuotes, DailyFactorValues, CorporateAction
#from selection_manager.service.selection_service import MARKET_INDICATOR_CODE
MARKET_INDICATOR_CODE='_MARKET_REGIME_INDICATOR_'
# [修改] 导入新的因子计算器
from selection_manager.service.stock_factor_calculator import StockFactorCalculator

logger = logging.getLogger(__name__)

FACTOR_LOOKBACK_BUFFER = 250 # 与prepare脚本保持一致

class StockValueService:
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    MODEL_FILE = MODELS_DIR / 'stock_lgbm_model.joblib'
    CONFIG_FILE = MODELS_DIR / 'stock_model_config.json'

    def __init__(self):
        self._model = None
        self._config = None
        self._feature_names = []
        self._dependencies_loaded = False
        self._load_dependencies()

    def _load_dependencies(self):
        if not self.MODEL_FILE.exists() or not self.CONFIG_FILE.exists():
            logger.error("个股评分模型或配置文件不存在。请先运行 'prepare_stock_features' 和 'train_stock_model'。")
            return
        try:
            self._model = joblib.load(self.MODEL_FILE)
            with open(self.CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            self._feature_names = self._config['feature_names']
            logger.info("成功加载个股评分模型及配置。")
            self._dependencies_loaded = True
        except Exception as e:
            logger.error(f"加载个股评分模型依赖时出错: {e}", exc_info=True)

    def get_all_stock_scores(self, stock_pool: list, trade_date, m_value: float, preloaded_panels: dict = None) -> pd.Series:
        if not self._dependencies_loaded:
            logger.warning("模型未加载，返回空评分列表。")
            return pd.Series(dtype=float)

        if preloaded_panels:
            logger.debug("StockValueService 检测到预加载面板数据，直接使用。")
            panel_data = preloaded_panels.copy()
            # 回测时，turnover列可能需要重命名
            if 'turnover' in panel_data and 'amount' not in panel_data:
                panel_data['amount'] = panel_data.pop('turnover')
        else:
            logger.info("未提供预加载面板，StockValueService 将从数据库加载实时数据...")
            panel_data = self._load_data_for_live(stock_pool, trade_date)

        if not panel_data or panel_data['close'].empty:
            logger.warning("数据加载后，面板为空，无法进行评分。")
            return pd.Series(dtype=float)

        # 启动向量化因子计算引擎
        logger.info("启动因子计算引擎...")
        calculator = StockFactorCalculator(panel_data, self._feature_names)
        features_df = calculator.run()

        if features_df.empty:
            logger.warning("特征计算后无有效数据。")
            return pd.Series(dtype=float)

        # 准备模型输入
        logger.info("准备模型输入并进行预测...")
        # 截取最后一天的特征
        latest_features = features_df.loc[pd.to_datetime(trade_date)]
        
        # 加入当日M值
        latest_features['market_m_value'] = m_value
        latest_features.dropna(inplace=True)
        
        if latest_features.empty:
            logger.warning("所有股票在特征计算后都因NaN被剔除。")
            return pd.Series(dtype=float)

        # 保证特征顺序和数据类型
        model_input = latest_features[self._feature_names].astype(float)
        
        # 一次性预测所有股票
        scores = self._model.predict(model_input)
        
        final_scores = pd.Series(scores, index=model_input.index)
        
        logger.info(f"成功为 {len(final_scores)} 只股票计算了模型评分。")
        return final_scores

    def _load_data_for_live(self, stock_pool: list, trade_date) -> dict:
        """在实时模式下，为给定的股票池和日期加载所有需要的数据并构建面板。"""
        if not stock_pool:
            return {}

        # 1. 定义数据加载的时间窗口
        start_date = trade_date - timedelta(days=FACTOR_LOOKBACK_BUFFER * 2) # 넉넉하게
        
        # 2. 批量加载行情数据
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool,
            trade_date__gte=start_date,
            trade_date__lte=trade_date
        ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close')
        
        if not quotes_qs:
            return {}
        
        quotes_df = pd.DataFrame.from_records(quotes_qs)
        quotes_df.rename(columns={'stock_code_id': 'stock_code', 'turnover': 'amount'}, inplace=True)

        # 3. 复权处理
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'hfq_close']
        for col in numeric_cols:
            quotes_df[col] = pd.to_numeric(quotes_df[col], errors='coerce')
        
        quotes_df['adj_factor'] = quotes_df['hfq_close'] / (quotes_df['close'] + self.epsilon)
        quotes_df['open'] = quotes_df['open'] * quotes_df['adj_factor']
        quotes_df['high'] = quotes_df['high'] * quotes_df['adj_factor']
        quotes_df['low'] = quotes_df['low'] * quotes_df['adj_factor']
        quotes_df['close'] = quotes_df['hfq_close']
        
        # 4. 构建基础价格面板
        panel_data = {}
        quotes_df['trade_date'] = pd.to_datetime(quotes_df['trade_date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            panel = quotes_df.pivot(index='trade_date', columns='stock_code', values=col)
            if not panel.empty:
                full_date_range = pd.date_range(start=panel.index.min(), end=panel.index.max(), freq='B')
                panel = panel.reindex(full_date_range).ffill()
            panel_data[col] = panel

        # 5. 加载并准备M值序列
        m_values_qs = DailyFactorValues.objects.filter(
            stock_code_id=MARKET_INDICATOR_CODE,
            factor_code_id='dynamic_M_VALUE',
            trade_date__gte=start_date,
            trade_date__lte=trade_date
        ).values('trade_date', 'raw_value')
        m_values_df = pd.DataFrame.from_records(m_values_qs)
        if not m_values_df.empty:
            m_values_df['trade_date'] = pd.to_datetime(m_values_df['trade_date'])
            panel_data['m_value_series'] = m_values_df.set_index('trade_date')['raw_value'].astype(float)
        else:
            panel_data['m_value_series'] = pd.Series(dtype=float)

        # 6. 加载并构建公司行动因子面板
        corp_actions_qs = CorporateAction.objects.filter(
            stock_code__in=stock_pool,
            ex_dividend_date__gte=start_date,
            ex_dividend_date__lte=trade_date,
            event_type__in=['dividend', 'bonus', 'transfer']
        ).values()
        corp_actions_df = pd.DataFrame.from_records(corp_actions_qs)
        
        # 复用prepare脚本中的构建逻辑
        from ..management.commands.prepare_stock_features import Command as PrepareCommand
        prepare_command_instance = PrepareCommand()
        panel_data['corp_action_panel'] = prepare_command_instance._build_corp_action_panel(corp_actions_df, panel_data['close'])

        return panel_data
