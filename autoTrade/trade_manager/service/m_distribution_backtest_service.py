# trade_manager/service/m_distribution_backtest_service.py (替换全部内容)

import logging
from datetime import date, timedelta, datetime
from decimal import Decimal

import pandas as pd
from django.db import connections, transaction
from django.core.management import call_command
from django.utils import timezone

from common.models import (
    DailyQuotes, StockInfo, CorporateAction, FactorDefinitions, StrategyParameters,
    DailyFactorValues, IndexQuotesCsi300, Position, TradeLog, DailyTradingPlan
)
from common.models.backtest_logs import MDistributionBacktestLog
from selection_manager.service.selection_service import SelectionService
from trade_manager.service.decision_order_service import DecisionOrderService
from .db_utils import use_backtest_schema
from .m_distribution_reporter import MDistributionReporter
from trade_manager.service.simulate_trade import SimulateTradeService
logger = logging.getLogger(__name__)

class MDistributionBacktestService:
    """
    M值胜率分布回测服务 (V2 - 修正版)。
    
    核心流程:
    1. 创建独立的数据库schema进行回测，与主环境隔离。
    2. 按天循环，每天运行SelectionService生成并保存交易预案（包含策略DNA）。
    3. 从数据库读取预案，对每个预案进行“前向追溯”。
    4. 将每次模拟交易的结果记录到专用的日志表中。
    5. 回测结束后，调用报告模块生成分析报告。
    """
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        self.backtest_run_id = f"m_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_holding_days = 90

    def _setup_backtest_schema(self):
        """
        【最终修正版】在 Schema 中准备回测环境，并集成索引/约束优化逻辑。
        """
        logger.info(f"--- 1. 在 Schema '{self.backtest_run_id}' 中准备回测环境 (集成索引优化) ---")
        
        with connections['default'].cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.backtest_run_id}";')
            cursor.execute(f'SET search_path TO "{self.backtest_run_id}";')
            call_command('migrate')
        logger.info("表结构创建完成。")
        tables_to_copy = [
            StockInfo, DailyQuotes, CorporateAction, FactorDefinitions,
            StrategyParameters, DailyFactorValues, IndexQuotesCsi300
        ]
        
        logger.info(f"准备从 'public' schema 复制基础数据到 '{self.backtest_run_id}'...")
        with transaction.atomic(), connections['default'].cursor() as cursor:
            # 确保后续操作都在新schema下
            cursor.execute(f'SET search_path TO "{self.backtest_run_id}";')
            
            for model in tables_to_copy:
                table_name = model._meta.db_table
                logger.info(f"  - 正在处理表: {table_name}")
                
                # =========================================================================
                # 1. 获取并暂存索引和约束的定义
                # =========================================================================
                # 1a. 获取普通索引 (不包括由UNIQUE或PRIMARY KEY约束创建的索引)
                cursor.execute("""
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = %s AND tablename = %s
                    AND indexname NOT IN (
                        SELECT conname FROM pg_constraint WHERE conrelid = %s::regclass
                    );
                """, [self.backtest_run_id, table_name, f'"{self.backtest_run_id}"."{table_name}"'])
                plain_indexes_to_recreate = [row[0] for row in cursor.fetchall()]
                # 1b. 获取约束 (外键、唯一、主键等)
                cursor.execute("""
                    SELECT 'ALTER TABLE ' || quote_ident(conrelid::regclass::text) || ' ADD CONSTRAINT ' || quote_ident(conname) || ' ' || pg_get_constraintdef(oid)
                    FROM pg_constraint
                    WHERE conrelid = %s::regclass;
                """, [f'"{self.backtest_run_id}"."{table_name}"'])
                constraints_to_recreate = [row[0] for row in cursor.fetchall()]
                
                # =========================================================================
                # 2. 删除所有约束和索引以极大地加速数据插入
                # =========================================================================
                for const_def in constraints_to_recreate:
                    const_name = const_def.split('ADD CONSTRAINT ')[1].split(' ')[0]
                    logger.debug(f"      - 删除约束: {const_name}")
                    cursor.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS {const_name} CASCADE;')
                for index_def in plain_indexes_to_recreate:
                    try:
                        index_name = index_def.split(' ')[2]
                        logger.debug(f"      - 删除索引: {index_name}")
                        cursor.execute(f'DROP INDEX IF EXISTS "{index_name}";')
                    except IndexError:
                        logger.warning(f"无法从 '{index_def}' 解析索引名称，跳过删除。")
                # =========================================================================
                # 3. 高效复制数据
                # =========================================================================
                logger.info(f"    - 正在从 public.{table_name} 复制数据...")
                sql = f'INSERT INTO "{table_name}" SELECT * FROM public."{table_name}";'
                cursor.execute(sql)
                logger.info(f"    - 数据复制完成。")
                # =========================================================================
                # 4. 重建索引和约束
                # =========================================================================
                logger.info(f"    - 正在重建 '{table_name}' 的索引和约束...")
                # 4a. 重建普通索引
                for index_def in plain_indexes_to_recreate:
                    logger.debug(f"      - 重建索引: {index_def}")
                    cursor.execute(index_def)
                
                # 4b. 重建约束 (这会自动重建它们的底层索引)
                #     注意：主键约束必须最先重建
                constraints_to_recreate.sort(key=lambda x: 'PRIMARY KEY' not in x)
                for const_def in constraints_to_recreate:
                    logger.debug(f"      - 重建约束: {const_def}")
                    cursor.execute(const_def)
                # =========================================================================
                # 5. 重置自增主键序列 (如果存在)
                # =========================================================================
                find_serial_sql = """
                    SELECT a.attname, pg_get_serial_sequence(quote_ident(n.nspname) || '.' || quote_ident(c.relname), a.attname)
                    FROM pg_class c
                    JOIN pg_attribute a ON a.attrelid = c.oid
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped
                    AND pg_get_serial_sequence(quote_ident(n.nspname) || '.' || quote_ident(c.relname), a.attname) IS NOT NULL;
                """
                cursor.execute(find_serial_sql, [self.backtest_run_id, table_name])
                serial_columns = cursor.fetchall()
                for column_name, sequence_name in serial_columns:
                    if sequence_name:
                        logger.info(f"    - 发现自增列 '{column_name}'，正在重置其序列 '{sequence_name}'...")
                        update_sequence_sql = f"""
                            SELECT setval('{sequence_name}', COALESCE((SELECT MAX("{column_name}") FROM "{table_name}"), 0) + 1, false);
                        """
                        cursor.execute(update_sequence_sql)
        
        logger.info("基础数据复制完成，并已完成索引优化。")

    def run(self):
        try:
            with use_backtest_schema(self.backtest_run_id):
                self._setup_backtest_schema()
                
                trading_days = list(DailyQuotes.objects.filter(
                    trade_date__gte=self.start_date,
                    trade_date__lte=self.end_date
                ).values_list('trade_date', flat=True).distinct().order_by('trade_date'))

                logger.info(f"--- 2. 开始日度回测循环 ({len(trading_days)}天) ---")
                
                for i, t_minus_1 in enumerate(trading_days):
                    logger.info(f"\n{'='*20} 模拟预案日: {t_minus_1} ({i+1}/{len(trading_days)}) {'='*20}")

                    # 1. 运行选股服务，它会自动将预案（含策略DNA）保存到数据库
                    selection_service = SelectionService(trade_date=t_minus_1, mode='backtest')
                    selection_service.run_selection()

                    # 2. 从数据库中查询刚刚生成的预案
                    plan_date_for_t = t_minus_1 + timedelta(days=1)
                    plans_for_today = DailyTradingPlan.objects.filter(plan_date=plan_date_for_t)

                    if not plans_for_today.exists():
                        logger.info(f"在 {t_minus_1} 未生成任何交易预案。")
                        continue
                    
                    # 3. 对每个从数据库读出的预案进行前向追溯
                    for plan in plans_for_today:
                        self._trace_forward_and_log(t_minus_1, plan, selection_service.market_regime_M)

            logger.info("--- 3. 回测循环结束, 生成报告 ---")
            reporter = MDistributionReporter(self.backtest_run_id)
            reporter.generate_and_send_report()

        except Exception as e:
            logger.critical(f"M值分布回测过程中发生严重错误: {e}", exc_info=True)
        finally:
            # 清理schema（可以暂时注释掉以便调试）
            # with connections['default'].cursor() as cursor:
            #     cursor.execute(f'DROP SCHEMA IF EXISTS "{self.backtest_run_id}" CASCADE;')
            # logger.info(f"已清理回测环境 Schema: {self.backtest_run_id}")
            pass

    def _trace_forward_and_log(self, t_minus_1: date, plan_obj: DailyTradingPlan, m_value: float):
        """对单个预案 (数据库对象) 进行前向追溯并记录结果"""
        stock_code = plan_obj.stock_code_id
        logger.debug(f"  -> 开始追溯股票: {stock_code}")

        try:
            entry_day_quote = DailyQuotes.objects.get(stock_code_id=stock_code, trade_date=plan_obj.plan_date)
            entry_date = entry_day_quote.trade_date
            entry_price = entry_day_quote.open
        except DailyQuotes.DoesNotExist:
            logger.warning(f"    无法找到 {plan_obj.plan_date} 的行情数据，无法为 {stock_code} 入场。")
            return

        try:
            tp_price, sl_price, tp_rate, sl_rate = self._get_simulated_stop_profit_loss(stock_code, entry_date, entry_price)
        except ValueError as e:
            logger.warning(f"    无法为 {stock_code} 计算止盈止损: {e}，跳过此股票。")
            return

        future_quotes_qs = DailyQuotes.objects.filter(
            stock_code_id=stock_code,
            trade_date__gt=entry_date
        ).order_by('trade_date').values('trade_date', 'high', 'low', 'close')[:self.max_holding_days]
        
        future_quotes = list(future_quotes_qs)
        if not future_quotes:
            logger.warning(f"    {stock_code} 在入场后无后续行情数据，无法追溯。")
            return

        exit_info = None
        for i, quote_dict in enumerate(future_quotes):
            if quote_dict['high'] >= tp_price:
                exit_info = {'date': quote_dict['trade_date'], 'price': tp_price, 'reason': 'TAKE_PROFIT', 'period': i + 1}
                break
            if quote_dict['low'] <= sl_price:
                exit_info = {'date': quote_dict['trade_date'], 'price': sl_price, 'reason': 'STOP_LOSS', 'period': i + 1}
                break
        
        if not exit_info:
            last_quote = future_quotes[-1]
            exit_info = {'date': last_quote['trade_date'], 'price': last_quote['close'], 'reason': 'END_OF_PERIOD', 'period': len(future_quotes)}
        
        actual_return = (exit_info['price'] / entry_price) - 1 if entry_price > 0 else 0

        MDistributionBacktestLog.objects.create(
            backtest_run_id=self.backtest_run_id,
            plan_date=t_minus_1,
            stock_code=stock_code,
            stock_name=plan_obj.stock_code.stock_name,
            m_value_at_plan=Decimal(str(m_value)),
            strategy_dna=plan_obj.strategy_dna,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=exit_info['date'],
            exit_price=exit_info['price'],
            exit_reason=exit_info['reason'],
            holding_period=exit_info['period'],
            preset_take_profit_rate=tp_rate,
            preset_stop_loss_rate=sl_rate,
            actual_return_rate=actual_return
        )
        logger.info(f"    [记录成功] {stock_code}: 入场 {entry_date}@{entry_price:.2f}, 出场 {exit_info['date']}@{exit_info['price']:.2f}, 原因: {exit_info['reason']}")

    def _get_simulated_stop_profit_loss(self, stock_code: str, entry_date: date, entry_price: Decimal):
        """复用DecisionOrderService逻辑计算止盈止损，但不实际修改数据库"""
        tp_price, sl_price, tp_rate, sl_rate = (Decimal(0), Decimal(0), Decimal(0), Decimal(0))
        
        with transaction.atomic():
            temp_position = Position.objects.create(
                stock_code_id=stock_code, entry_price=entry_price, quantity=100,
                entry_datetime=timezone.now(), status=Position.StatusChoices.OPEN,
                current_stop_loss=Decimal('0.00'),
                current_take_profit=Decimal('0.00')
            )
            temp_trade_log = TradeLog.objects.create(
                position=temp_position, stock_code_id=stock_code, trade_datetime=timezone.now(),
                trade_type=TradeLog.TradeTypeChoices.BUY, status=TradeLog.StatusChoices.FILLED,
                price=entry_price, quantity=100,
                commission=0,
                stamp_duty=0
            )

            decision_service = DecisionOrderService(handler=None, execution_date=entry_date)
            decision_service.calculate_stop_profit_loss(trade_id=temp_trade_log.trade_id)

            temp_position.refresh_from_db()
            tp_price = temp_position.current_take_profit
            sl_price = temp_position.current_stop_loss
            
            if entry_price > 0:
                tp_rate = (tp_price / entry_price) - 1
                sl_rate = 1 - (sl_price / entry_price)

            transaction.set_rollback(True)
            
        if tp_price == 0 or sl_price == 0:
            raise ValueError("计算出的止盈止损价无效")
            
        return tp_price, sl_price, tp_rate, sl_rate

