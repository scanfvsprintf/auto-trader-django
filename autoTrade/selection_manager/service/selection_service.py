# ==============================================================================
# 文件 4/4: selection_manager/service/selection_service.py (重构版)
# 描述: 简化后的选股服务，职责分离，调用新服务获取评分。
# ==============================================================================
import logging
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
from django.db import transaction
from common.models import IndexQuotesCsi300
from common.models import (
    StockInfo, DailyQuotes, SystemLog, FactorDefinitions, DailyFactorValues,
    StrategyParameters, DailyTradingPlan
)
from .m_value_service import m_value_service_instance # 复用M值预测服务
from .stock_value_service import StockValueService # 引入新的个股评分服务

logger = logging.getLogger(__name__)
MODULE_NAME = '机器学习选股与预案模块'
MARKET_INDICATOR_CODE = '_MARKET_REGIME_INDICATOR_'

class SelectionService:
    """
    [重构版] T-1日收盘后运行的，基于机器学习模型的选股与预案生成服务。
    
    该服务现在将核心的股票评分逻辑委托给 StockValueService，自身职责简化为：
    1. 计算市场状态 M(t)。
    2. 调用 StockValueService 获取所有股票的模型评分。
    3. 基于模型评分生成交易预案。
    4. 保存结果。
    """

    def __init__(self, trade_date: date, mode: str = 'realtime', one_strategy: str = None, preloaded_panels: dict = None):
        if mode not in ['realtime', 'backtest']:
            raise ValueError("模式(mode)必须是 'realtime' 或 'backtest'")

        try:
            latest_trade_date_obj = DailyQuotes.objects.filter(trade_date__lte=trade_date).latest('trade_date')
            self.trade_date = latest_trade_date_obj.trade_date
        except DailyQuotes.DoesNotExist:
            self.trade_date = trade_date
        
        self.mode = mode
        self.params = {}
        self.market_regime_M = 0.0
        self.stock_value_service = StockValueService() # 实例化新的服务
        self.preloaded_panels = preloaded_panels
        logger.debug(f"--- {MODULE_NAME} 初始化 ---")
        logger.debug(f"交易日期 (T-1): {self.trade_date}, 运行模式: {self.mode}")

    def run_selection(self):
        """一键启动全流程的入口方法。"""
        self._log_to_db('INFO', f"机器学习选股流程启动。模式: {self.mode}, 日期: {self.trade_date}")
        try:
            self._load_parameters()
            initial_stock_pool = self._initial_screening()
            if not initial_stock_pool:
                self._log_to_db('WARNING', "初步筛选后无符合条件的股票，流程终止。")
                return

            # 1. 计算市场M值
            self.market_regime_M = self._calculate_market_regime_M(initial_stock_pool)

            # 2. 调用新服务获取所有股票的模型评分
            logger.info("调用StockValueService获取所有股票的模型评分...")
            final_scores = self.stock_value_service.get_all_stock_scores(
                stock_pool=initial_stock_pool,
                trade_date=self.trade_date,
                m_value=self.market_regime_M,
                preloaded_panels=self.preloaded_panels
            )
            final_scores = final_scores.sort_values(ascending=False)

            if final_scores.empty:
                self._log_to_db('WARNING', "模型未对任何股票给出有效评分，流程终止。")
                return
            if self.preloaded_panels:
                plan_panels = self.preloaded_panels
            else:
                # 如果是实时运行，没有预加载数据，需要为TopN股票加载数据
                top_n_codes = final_scores.head(int(self.params.get('dynamic_top_n', 10))).index.tolist()
                plan_panels = self._load_panels_for_plan(top_n_codes)
            trading_plan = self._generate_trading_plan(final_scores, plan_panels)
            # 3. 生成交易预案
            #trading_plan = self._generate_trading_plan(final_scores)
            if trading_plan.empty:
                self._log_to_db('WARNING', "最终未生成任何交易预案。")
                return

            # 4. 保存结果
            self._save_results(final_scores, trading_plan)

            success_msg = f"机器学习选股流程成功完成。M(t)={self.market_regime_M:.4f}, 生成 {len(trading_plan)} 条交易预案。"
            logger.info(success_msg)
            self._log_to_db('INFO', success_msg)

        except Exception as e:
            error_msg = f"机器学习选股流程发生严重错误: {e}"
            logger.critical(error_msg, exc_info=True)
            self._log_to_db('CRITICAL', error_msg)
            raise

    def _load_parameters(self):
        """加载策略参数"""
        logger.debug("加载策略参数...")
        params_qs = StrategyParameters.objects.all()
        self.params = {p.param_name: float(p.param_value) for p in params_qs}
        
    def _initial_screening(self) -> list[str]:
        """初步筛选股票池，逻辑保持不变"""
        logger.debug("开始执行初步筛选...")
        all_stocks = StockInfo.objects.filter(status=StockInfo.StatusChoices.LISTING)
        non_st_stocks = all_stocks.exclude(stock_code__contains='.688').exclude(stock_name__startswith='ST').exclude(stock_name__startswith='*ST')
        
        min_listing_days = self.params.get('dynamic_lookback_new_stock', 60)
        min_listing_date = self.trade_date - timedelta(days=int(min_listing_days))
        non_new_stocks = non_st_stocks.filter(listing_date__lt=min_listing_date)
        
        stock_pool_codes = list(non_new_stocks.values_list('stock_code', flat=True))
        
        lookback_days = 20
        start_date = self.trade_date - timedelta(days=lookback_days * 2)
        quotes = DailyQuotes.objects.filter(
            stock_code_id__in=stock_pool_codes,
            trade_date__gte=start_date,
            trade_date__lte=self.trade_date
        ).values('stock_code_id', 'turnover')

        if not quotes:
            return []

        quotes_df = pd.DataFrame.from_records(quotes)
        avg_turnover = quotes_df.groupby('stock_code_id')['turnover'].mean()
        
        min_liquidity = self.params.get('dynamic_min_liquidity', 100000000)
        liquid_stocks = avg_turnover[avg_turnover >= min_liquidity]
        
        final_stock_pool = list(liquid_stocks.index)
        logger.debug(f"初步筛选后，最终剩余 {len(final_stock_pool)} 只股票进入精选池。")
        return final_stock_pool

    def _calculate_market_regime_M(self, stock_pool: list[str]) -> float:
        """
        计算市场状态函数 M(t)
        """
        
        # =======================================================================
        # [ML预测接口] - 未来切换到机器学习模型预测M值的入口
        # =======================================================================
        try:
            # 1. 获取最近60个交易日的数据
            quotes_60_days_qs = IndexQuotesCsi300.objects.filter(
                trade_date__lte=self.trade_date
            ).order_by('-trade_date')[:100]
            
            if len(quotes_60_days_qs) < 100:
                logger.warning("沪深300数据不足100天，无法使用ML模型进行预测，将回退到传统方法。")
            else:
                # [修复] 从QuerySet直接构建DataFrame
                df_60_days_raw = pd.DataFrame.from_records(quotes_60_days_qs.values())
                
                # 反转顺序使日期从旧到新
                df_60_days = df_60_days_raw.iloc[::-1].reset_index(drop=True)
                
                # 2. 调用预测服务 (m_value_service内部会处理类型转换)
                ml_m_value = m_value_service_instance.predict_csi300_next_day_trend(df_60_days)
                
                # 3. 【重要】将ML预测结果存入缓存
                DailyFactorValues.objects.update_or_create(
                    stock_code_id=MARKET_INDICATOR_CODE,
                    trade_date=self.trade_date,
                    factor_code_id='dynamic_M_VALUE',
                    defaults={'raw_value': Decimal(str(ml_m_value)), 'norm_score': Decimal(str(ml_m_value))}
                )
                logger.info(f"已使用ML模型预测M(t) = {ml_m_value:.4f}")
                return ml_m_value
        except Exception as e:
            logger.error(f"调用ML模型预测M值时发生错误: {e}", exc_info=True)

    def _generate_trading_plan(self, final_scores: pd.Series, panels: dict) -> pd.DataFrame:
        """基于模型评分和面板数据生成交易预案"""
        logger.debug("基于模型评分生成交易预案...")
        top_n = int(self.params.get('dynamic_top_n', 10))
        top_stocks_scores = final_scores.head(top_n)
        
        if top_stocks_scores.empty:
            return pd.DataFrame()
        top_stock_codes = top_stocks_scores.index.tolist()
        
        # --- [关键修正] 开始 ---
        # 不再从数据库加载，直接使用传入的panels
        if not panels or 'close' not in panels:
            logger.error("生成交易预案时未提供有效的面板数据。")
            return pd.DataFrame()
        # 注意：MIOP/MAOP的计算应该基于不复权价格，但ATR也需要不复权价格
        # 我们需要确保传入的panels包含不复权价格。
        # 假设回测框架传入的panels已经是处理好的，包含了所需列。
        # 如果没有，我们需要在这里进行处理，但为了保持与你原设计一致，我们假设panels是OK的。
        # 这里的逻辑需要和你的回测框架传入的数据格式对齐。
        # 假设你的回测框架传入的panels的 'close', 'high', 'low' 是不复权价。
        close_panel = panels['close']
        high_panel = panels['high']
        low_panel = panels['low']
        last_close_series = close_panel.iloc[-1].reindex(top_stock_codes)
        # 简单的日内波幅作为ATR
        last_atr_series = (high_panel.iloc[-1] - low_panel.iloc[-1]).reindex(top_stock_codes)
        # --- [关键修正] 结束 ---
        k_gap = self.params.get('k_gap', 0.5)
        k_drop = self.params.get('k_drop', 0.3)
        plans = []
        for stock_code, score in top_stocks_scores.items():
            close_price = last_close_series.get(stock_code)
            atr = last_atr_series.get(stock_code)
            if pd.isna(close_price) or pd.isna(atr):
                continue
            
            miop = Decimal(str(close_price)) - Decimal(str(k_drop)) * Decimal(str(atr))
            maop = Decimal(str(close_price)) + Decimal(str(k_gap)) * Decimal(str(atr))
            plans.append({
                'stock_code': stock_code,
                'rank': len(plans) + 1,
                'final_score': score,
                'miop': miop,
                'maop': maop,
            })
        
        return pd.DataFrame(plans)
    

    def _load_panels_for_plan(self, stock_codes: list) -> dict:
        """在实时模式下，为生成交易预案加载所需的不复权价格面板"""
        if not stock_codes:
            return {}
        
        start_date = self.trade_date - timedelta(days=30) # ATR计算通常不需要很长回溯
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=stock_codes,
            trade_date__gte=start_date,
            trade_date__lte=self.trade_date
        ).values('trade_date', 'stock_code_id', 'close', 'high', 'low')
        if not quotes_qs:
            return {}
            
        df = pd.DataFrame.from_records(quotes_qs)
        panels = {}
        for col in ['close', 'high', 'low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            panel = df.pivot(index='trade_date', columns='stock_code_id', values=col)
            panels[col] = panel
        return panels

    @transaction.atomic
    def _save_results(self, final_scores: pd.Series, trading_plan_df: pd.DataFrame):
        """保存模型评分和交易预案"""
        logger.debug("开始将结果保存到数据库...")

        # 1. 保存模型评分到 DailyFactorValues
        factor_values_to_create = []
        for stock_code, score in final_scores.items():
            factor_values_to_create.append(
                DailyFactorValues(
                    stock_code_id=stock_code,
                    trade_date=self.trade_date,
                    factor_code_id='ML_STOCK_SCORE', # 新因子
                    raw_value=Decimal(str(score)),
                    norm_score=Decimal(str(score)) # 评分本身就是标准化的，直接存
                )
            )
        
        # 先删除当日旧的ML_STOCK_SCORE，再批量创建
        DailyFactorValues.objects.filter(
            trade_date=self.trade_date,
            factor_code_id='ML_STOCK_SCORE'
        ).delete()
        DailyFactorValues.objects.bulk_create(factor_values_to_create, batch_size=1000)
        logger.debug(f"已保存 {len(factor_values_to_create)} 条个股模型评分。")

        # 2. 保存交易预案
        plan_date = self.trade_date + timedelta(days=1)
        DailyTradingPlan.objects.filter(plan_date=plan_date).delete()
        
        plans_to_create = []
        for _, row in trading_plan_df.iterrows():
            plans_to_create.append(
                DailyTradingPlan(
                    plan_date=plan_date, stock_code_id=row['stock_code'],
                    rank=row['rank'], final_score=Decimal(str(row['final_score'])),
                    miop=row['miop'].quantize(Decimal('0.01')),
                    maop=row['maop'].quantize(Decimal('0.01')),
                    status=DailyTradingPlan.StatusChoices.PENDING,
                    strategy_dna="ML_MODEL:1.00" # 策略DNA现在固定为模型
                )
            )
        DailyTradingPlan.objects.bulk_create(plans_to_create)
        logger.debug(f"已保存 {len(plans_to_create)} 条交易预案。")

    def _log_to_db(self, level, message):
        logger.info(message)
        # 在回测等高性能场景下可以关闭
        # SystemLog.objects.create(log_level=level, module_name=MODULE_NAME, message=message)
        pass

