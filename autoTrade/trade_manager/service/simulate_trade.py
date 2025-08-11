# trade_manager/service/simulate_trade.py

import logging
from datetime import date, timedelta, datetime
from decimal import Decimal
import numpy as np
import pandas as pd
from django.db import connections, transaction
from django.core.management import call_command

# 内部模块导入
from common.models import (
    DailyFactorValues, DailyTradingPlan, Position, TradeLog, SystemLog,
    StrategyParameters, DailyQuotes, CorporateAction
)
from selection_manager.service.selection_service import SelectionService
from trade_manager.service.before_fix_service import BeforeFixService
from trade_manager.service.decision_order_service import DecisionOrderService
from trade_manager.service.monitor_exit_service import MonitorExitService
from .simulate_trade_handler import SimulateTradeHandler
from .db_utils import use_backtest_schema  # <-- 导入新的工具

logger = logging.getLogger(__name__)

class SimulateTradeService:
    """
    回测实施服务 (V2 - PostgreSQL 动态 Schema 版)。
    """
    COMMISSION_RATE = Decimal('0.0002854')
    MIN_COMMISSION = Decimal('5')
    STAMP_DUTY_RATE = Decimal('0.001')
    SELL_SLIPPAGE_RATE = Decimal('0.002')
    ANNUAL_RISK_FREE_RATE = Decimal('0.015')
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self):
        self.start_date: date = None
        self.end_date: date = None
        self.current_date: date = None
        self.initial_capital = Decimal('0.0')
        self.cash_balance = Decimal('0.0')
        self.portfolio_history = []
        self.last_buy_trade_id = None
        # self.original_db_config 不再需要

    def _setup_backtest_schema(self, schema_name: str):
        """
        在新 schema 中创建表结构并复制基础数据。
        此函数必须在 `use_backtest_schema` 上下文中调用。
        """
        logger.info(f"--- 1. 在 Schema '{schema_name}' 中准备回测环境 ---")
        
        # 1. 创建所有表结构
        logger.info("正在新 Schema 中创建表结构 (执行 migrate)...")
        call_command('migrate')
        logger.info("表结构创建完成。")

        # 2. 定义需要从 public schema 复制的基础数据表
        tables_to_copy = [
            'tb_stock_info', 'tb_daily_quotes', 'tb_corporate_actions',
            'tb_factor_definitions', 'tb_strategy_parameters', 'tb_daily_factor_values'
        ]
        
        logger.info(f"准备从 'public' schema 复制基础数据到 '{schema_name}'...")
        with connections['default'].cursor() as cursor:
            for table_name in tables_to_copy:
                logger.info(f"  - 正在复制表: {table_name}")
                # 使用 INSERT INTO ... SELECT * ... 高效复制数据
                sql = f'INSERT INTO "{schema_name}"."{table_name}" SELECT * FROM public."{table_name}";'
                cursor.execute(sql)
        logger.info("基础数据复制完成。")

        # 3. 初始化资金
        params = {p.param_name: p.param_value for p in StrategyParameters.objects.all()}
        max_positions = int(params.get('MAX_POSITIONS', Decimal('5')))
        max_capital_per_pos = params.get('MAX_CAPITAL_PER_POSITION', Decimal('10000'))
        self.initial_capital = Decimal('150000') # 按你要求写死
        self.cash_balance = self.initial_capital
        logger.info(f"初始资金已设定为: {self.initial_capital:.2f}")

    def run_backtest(self, start_date: str, end_date: str) -> dict:
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)

        # 1. 生成唯一的 schema 名称
        schema_name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"为本次回测创建临时 Schema: {schema_name}")

        try:
            # 2. 创建 Schema
            with connections['default'].cursor() as cursor:
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";')

            # 3. 进入动态 Schema 上下文
            with use_backtest_schema(schema_name):
                # 3.1 在新 schema 中准备环境（建表、复制数据）
                self._setup_backtest_schema(schema_name)

                # 3.2 执行回测主循环 (这部分逻辑基本不变)
                handler = SimulateTradeHandler(self)
                trading_days = self._get_trading_days()
                if not trading_days:
                    logger.error("在指定日期范围内未找到任何交易日，回测终止。")
                    return {}

                baseline_date = trading_days[0] - timedelta(days=1)
                self.portfolio_history.append({'date': baseline_date, 'total_value': self.initial_capital})

                logger.info(f"--- 2. 开始日度回测循环 ({len(trading_days)}天) ---")
                for i, current_day in enumerate(trading_days):
                    self.current_date = current_day
                    logger.info(f"\n{'='*20} 模拟日: {self.current_date} ({i+1}/{len(trading_days)}) {'='*20}")

                    # ... (原有的回测循环逻辑，无需修改) ...
                    prev_trading_day = trading_days[i-1] if i > 0 else None
                    if prev_trading_day:
                        logger.info(f"-> [T-1 选股] 基于 {prev_trading_day} 的数据...")
                        selection_service = SelectionService(trade_date=prev_trading_day, mode='backtest')
                        selection_service.run_selection()

                    logger.info("-> [T日 盘前校准] ...")
                    before_fix_service = BeforeFixService(execution_date=self.current_date)
                    before_fix_service.run()
                    
                    # ... (分红逻辑) ...
                    dividend_events = CorporateAction.objects.filter(
                        ex_dividend_date=self.current_date, event_type=CorporateAction.EventType.DIVIDEND
                    )
                    events_by_stock = {}
                    for event in dividend_events:
                        events_by_stock.setdefault(event.stock_code, []).append(event)
                    if events_by_stock:
                        open_positions_for_dividend = Position.objects.filter(
                            stock_code_id__in=events_by_stock.keys(),
                            status=Position.StatusChoices.OPEN
                        )
                        for pos in open_positions_for_dividend:
                            stock_events = events_by_stock.get(pos.stock_code_id, [])
                            for event in stock_events:
                                dividend_amount = event.dividend_per_share * pos.quantity
                                self.cash_balance += dividend_amount
                                logger.info(f"除息事件: 持仓ID {pos.position_id} ({pos.stock_code_id}) 获得分红 {dividend_amount:.2f}，现金余额更新为 {self.cash_balance:.2f}")

                    logger.info("-> [T日 开盘决策与买入] ...")
                    decision_order_service = DecisionOrderService(handler=handler, execution_date=self.current_date)
                    decision_order_service.adjust_trading_plan_daily()
                    
                    while True:
                        open_positions_count = Position.objects.filter(status=Position.StatusChoices.OPEN).count()
                        max_pos = decision_order_service.current_max_positions
                        if open_positions_count >= max_pos:
                            break
                        
                        self.last_buy_trade_id = None
                        decision_order_service.execute_orders()
                        
                        if self.last_buy_trade_id:
                            decision_order_service.calculate_stop_profit_loss(self.last_buy_trade_id)
                        else:
                            break

                    monitor_exit_service = MonitorExitService(handler=handler, execution_date=self.current_date)

                    logger.info("-> [T日 盘中监控] 模拟价格跌至最低点...")
                    handler.current_price_node = 'LOW'
                    monitor_exit_service.monitor_and_exit_positions()

                    logger.info("-> [T日 盘中监控] 模拟价格涨至最高点...")
                    handler.current_price_node = 'HIGH'
                    monitor_exit_service.monitor_and_exit_positions()

                    self._calculate_daily_portfolio_value()

                logger.info("--- 3. 回测循环结束 ---")
                return self._calculate_performance_metrics()

        except Exception as e:
            logger.critical(f"回测过程中发生严重错误: {e}", exc_info=True)
            return {"error": str(e)}
        # `finally` 块不再需要，因为上下文管理器会自动处理清理工作

    # _get_trading_days, _calculate_daily_portfolio_value, _calculate_performance_metrics 方法保持不变
    def _get_trading_days(self) -> list[date]:
        dates = DailyQuotes.objects.filter(
            trade_date__gte=self.start_date,
            trade_date__lte=self.end_date
        ).values_list('trade_date', flat=True).distinct().order_by('trade_date')
        return list(dates)

    def _calculate_daily_portfolio_value(self):
        open_positions = Position.objects.filter(status=Position.StatusChoices.OPEN)
        market_value = Decimal('0.0')
        for pos in open_positions:
            try:
                quote = DailyQuotes.objects.get(
                    stock_code_id=pos.stock_code_id,
                    trade_date=self.current_date
                )
                market_value += quote.close * pos.quantity
            except DailyQuotes.DoesNotExist:
                market_value += pos.entry_price * pos.quantity
        
        total_value = self.cash_balance + market_value
        self.portfolio_history.append({
            'date': self.current_date,
            'total_value': total_value
        })
        logger.info(f"--- 日终结算 ({self.current_date}) ---")
        logger.info(f"现金: {self.cash_balance:.2f}, 持仓市值: {market_value:.2f}, 总资产: {total_value:.2f}")

    def _calculate_performance_metrics(self) -> dict:
        logger.info("--- 4. 计算回测性能指标 ---")
        if not self.portfolio_history:
            return {}

        df = pd.DataFrame(self.portfolio_history)
        df['total_value'] = df['total_value'].astype(float)
        df['daily_return'] = df['total_value'].pct_change().fillna(0)
        
        final_value = float(df['total_value'].iloc[-1])
        total_return_amount = final_value - float(self.initial_capital)
        total_return_rate = (final_value / float(self.initial_capital)) - 1

        mean_daily_return = df['daily_return'].mean()
        std_daily_return = df['daily_return'].std()
        
        if std_daily_return == 0 or np.isnan(std_daily_return):
            sharpe_ratio = 0.0
        else:
            daily_risk_free_rate = (1 + self.ANNUAL_RISK_FREE_RATE) ** Decimal(1/self.TRADING_DAYS_PER_YEAR) - 1
            sharpe_ratio = (mean_daily_return - float(daily_risk_free_rate)) / std_daily_return
            sharpe_ratio *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        result = {
            'total_return_amount': round(total_return_amount, 2),
            'total_return_rate': round(total_return_rate, 4),
            'sharpe_ratio': round(float(sharpe_ratio), 4)
        }
        logger.info(f"回测结果: {result}")
        return result
