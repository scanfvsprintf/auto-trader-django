# trade_manager/service/simulate_trade.py

import logging
import shutil
import os
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import time
from django.conf import settings
from django.db import connections, transaction
import sqlite3
from common.models import (
    DailyFactorValues, DailyTradingPlan, Position, TradeLog, SystemLog,
    StrategyParameters, DailyQuotes, CorporateAction
)
from selection_manager.service.selection_service import SelectionService
from trade_manager.service.before_fix_service import BeforeFixService
from trade_manager.service.decision_order_service import DecisionOrderService
from trade_manager.service.monitor_exit_service import MonitorExitService
from .simulate_trade_handler import SimulateTradeHandler

logger = logging.getLogger(__name__)

class SimulateTradeService:
    """
    回测实施服务。
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
        self.original_db_config = None
    def _load_db_to_memory(self, source_db_path: str):
        """
        【优化版】使用 SQLite Backup API 高效地将磁盘数据库加载到内存。
        """
        logger.info(f"开始将数据从 {source_db_path} 加载到内存 (使用 Backup API)...")
        start_time = time.time()
        
        # 1. 创建一个到源文件数据库的直接连接 (只读)
        try:
            source_conn = sqlite3.connect(f'file:{source_db_path}?mode=ro', uri=True)
        except sqlite3.OperationalError as e:
            logger.error(f"无法以只读模式打开源数据库 {source_db_path}: {e}")
            raise
 
        # 2. 获取到Django管理的内存数据库的底层连接
        mem_conn = connections['default'].connection
 
        try:
            # 3. 【核心优化】使用 backup 方法
            #    它会以最有效的方式（通常是按数据页）将源数据库内容复制到目标数据库
            source_conn.backup(mem_conn)
            
            duration = time.time() - start_time
            logger.info(f"数据成功加载到内存数据库，耗时: {duration:.2f} 秒。")
 
        except Exception as e:
            logger.error(f"使用 Backup API 加载数据到内存时发生错误: {e}")
            raise
        finally:
            # 4. 关闭连接
            source_conn.close()
            # mem_conn 不需要我们手动关闭，Django会管理它
 
    def _setup_environment(self):
        """
        修正版：调整了操作顺序，先加载数据，再执行ORM操作。
        """
        logger.info("--- 1. 准备回测环境 (内存模式) ---")
        
        base_dir = settings.BASE_DIR
        source_db = os.path.join(base_dir, 'mainDB.sqlite3')
        
        # 1. 关闭所有现有连接
        connections.close_all()
        
        # 2. 保存原始配置，并将 'default' 数据库重定向到内存
        self.original_db_config = settings.DATABASES['default'].copy()
        settings.DATABASES['default']['NAME'] = ':memory:'
        logger.info("已将 'default' 数据库连接重定向到 :memory:")
 
        # 3. 确保Django建立到新内存数据库的连接
        #    这一步至关重要，它会创建一个空的内存数据库实例
        connections['default'].ensure_connection()
        
        # 4. 【核心修正】立即将磁盘数据加载到内存数据库中
        #    此时，内存数据库从空变成了 mainDB.sqlite3 的一个完整克隆
        self._load_db_to_memory(source_db)
 
        # 5. 【顺序调整】现在内存数据库是完整的了，可以安全地执行任何Django ORM操作
        

        #DailyFactorValues, DailyTradingPlan,
        # 清空回测过程中会产生数据的表
        tables_to_clear = [
             Position,
            TradeLog, SystemLog
        ]
        # 使用 transaction.atomic() 来保证操作的原子性
        with transaction.atomic():
            for model in tables_to_clear:
                # 现在 model.objects.all() 可以正常工作了
                model.objects.all().delete()
                logger.info(f"已清空表: {model._meta.db_table}")
 
        # 读取策略参数
        # 现在 StrategyParameters.objects.all() 也可以正常工作了
        params = {p.param_name: p.param_value for p in StrategyParameters.objects.all()}
        max_positions = int(params.get('MAX_POSITIONS', Decimal('5')))
        max_capital_per_pos = params.get('MAX_CAPITAL_PER_POSITION', Decimal('10000'))
        self.initial_capital = Decimal(max_positions) * max_capital_per_pos
        self.initial_capital=150000
        self.cash_balance = self.initial_capital
        logger.info(f"初始资金已设定为: {self.initial_capital:.2f}")
 
    def _cleanup_environment(self):
        """在回测结束后恢复原始数据库配置"""
        if self.original_db_config:
            connections.close_all()
            settings.DATABASES['default'] = self.original_db_config
            # 内存数据库的连接关闭后，其内容会自动销毁，无需手动删除文件
            logger.info("已恢复 'default' 数据库连接到原始配置，内存数据库已释放。")
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

    def run_backtest(self, start_date: str, end_date: str) -> dict:
        try:
            self.start_date = date.fromisoformat(start_date)
            self.end_date = date.fromisoformat(end_date)

            self._setup_environment()

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

                prev_trading_day = trading_days[i-1] if i > 0 else None
                if prev_trading_day:
                    logger.info(f"-> [T-1 选股] 基于 {prev_trading_day} 的数据...")
                    selection_service = SelectionService(trade_date=prev_trading_day, mode='backtest')
                    selection_service.run_selection()

                logger.info("-> [T日 盘前校准] ...")
                before_fix_service = BeforeFixService(execution_date=self.current_date)
                before_fix_service.run()
                
                dividend_events = CorporateAction.objects.filter(
                    ex_dividend_date=self.current_date, event_type=CorporateAction.EventType.DIVIDEND
                )
                # 按股票代码分组，提高效率
                events_by_stock = {}
                for event in dividend_events:
                    events_by_stock.setdefault(event.stock_code, []).append(event)

                if events_by_stock:
                    # 获取所有可能受影响的持仓
                    open_positions_for_dividend = Position.objects.filter(
                        stock_code_id__in=events_by_stock.keys(),
                        status=Position.StatusChoices.OPEN
                    )
                    
                    for pos in open_positions_for_dividend:
                        # 找到该股票对应的所有分红事件（通常只有一个）
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

                # 关键修复：在循环内实例化 MonitorExitService 并传入日期
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
        
        finally:
            # 确保无论成功还是失败，都清理环境
            self._cleanup_environment()
