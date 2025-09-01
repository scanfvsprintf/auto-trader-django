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

    def __init__(self, trade_date: date, mode: str = 'realtime'):
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
            self.stdout.write("调用StockValueService获取所有股票的模型评分...")
            final_scores = self.stock_value_service.get_all_stock_scores(
                stock_pool=initial_stock_pool,
                trade_date=self.trade_date,
                m_value=self.market_regime_M
            )
            final_scores = final_scores.sort_values(ascending=False)

            if final_scores.empty:
                self._log_to_db('WARNING', "模型未对任何股票给出有效评分，流程终止。")
                return

            # 3. 生成交易预案
            trading_plan = self._generate_trading_plan(final_scores)
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

    def _generate_trading_plan(self, final_scores: pd.Series) -> pd.DataFrame:
        """基于模型评分生成交易预案"""
        logger.debug("基于模型评分生成交易预案...")
        top_n = int(self.params.get('dynamic_top_n', 10))
        top_stocks_scores = final_scores.head(top_n)
        
        if top_stocks_scores.empty:
            return pd.DataFrame()

        top_stock_codes = top_stocks_scores.index.tolist()
        
        # 获取计算MIOP/MAOP所需的数据
        quotes_qs = DailyQuotes.objects.filter(
            stock_code_id__in=top_stock_codes,
            trade_date__lte=self.trade_date
        ).order_by('-trade_date').values('stock_code_id', 'close', 'high', 'low')
        quotes_df = pd.DataFrame.from_records(quotes_qs)
        numeric_cols = ['close', 'high', 'low']
        for col in numeric_cols:
            quotes_df[col] = pd.to_numeric(quotes_df[col], errors='coerce')
        # 构建一个字典，方便快速查找
        latest_quotes = {}
        for q in quotes_qs:
            if q['stock_code_id'] not in latest_quotes:
                latest_quotes[q['stock_code_id']] = q

        k_gap = self.params.get('k_gap', 0.5)
        k_drop = self.params.get('k_drop', 0.3)

        plans = []
        for stock_code, score in top_stocks_scores.items():
            quote = latest_quotes.get(stock_code)
            if not quote:
                continue
            
            close_price = Decimal(str(quote['close']))
            # 简单ATR计算
            atr = Decimal(str(quote['high'])) - Decimal(str(quote['low']))
            
            miop = close_price - Decimal(str(k_drop)) * atr
            maop = close_price + Decimal(str(k_gap)) * atr

            plans.append({
                'stock_code': stock_code,
                'rank': len(plans) + 1,
                'final_score': score,
                'miop': miop,
                'maop': maop,
            })
        
        return pd.DataFrame(plans)

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
        # 在回测等高性能场景下可以关闭
        # SystemLog.objects.create(log_level=level, module_name=MODULE_NAME, message=message)
        pass

