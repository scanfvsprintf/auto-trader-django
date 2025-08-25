# ==============================================================================
# 文件 4/5: trade_manager/service/simulate_trade.py (修改)
# 描述: 核心回测服务，集成日志记录和邮件发送。
# ==============================================================================
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
# 新增导入
from common.models.backtest_logs import BacktestDailyLog, BacktestOperationLog 
from selection_manager.service.selection_service import SelectionService, MARKET_INDICATOR_CODE
from trade_manager.service.before_fix_service import BeforeFixService
from trade_manager.service.decision_order_service import DecisionOrderService
from trade_manager.service.monitor_exit_service import MonitorExitService
from .simulate_trade_handler import SimulateTradeHandler
from .db_utils import use_backtest_schema
from .backtest_reporter import BacktestReporter 
from .position_monitor_logic import PositionMonitorLogic

logger = logging.getLogger(__name__)

class SimulateTradeService:
    """
    回测实施服务 (V3 - 集成日志与报告)。
    """
    COMMISSION_RATE = Decimal('0.0002854')
    MIN_COMMISSION = Decimal('5')
    STAMP_DUTY_RATE = Decimal('0.001')
    SELL_SLIPPAGE_RATE = Decimal('0.002')

    def __init__(self):
        self.start_date: date = None
        self.end_date: date = None
        self.current_date: date = None
        self.initial_capital = Decimal('0.0')
        self.cash_balance = Decimal('0.0')
        self.portfolio_history = []
        self.last_buy_trade_id = None
        self.backtest_run_id: str = None # 新增：回测唯一ID
        self.params: dict = {}
    def _persist_risk_prices_if_changed(self, position, new_sl, new_tp, label='盘中'):
        if new_sl != position.current_stop_loss or new_tp != position.current_take_profit:
            position.current_stop_loss = new_sl
            position.current_take_profit = new_tp
            position.save(update_fields=['current_stop_loss', 'current_take_profit'])
            logger.info(f"[回测] {position.stock_code_id} {label}更新风控价格。SL: {position.current_stop_loss:.2f}, TP: {position.current_take_profit:.2f}")
    def _load_parameters(self):
        """在回测开始前加载所有需要的策略参数。"""
        logger.info("加载回测所需策略参数...")
        # 从 StrategyParameters 表中一次性加载所有参数
        all_params = {p.param_name: p.param_value for p in StrategyParameters.objects.all()}
        
        # 定义回测逻辑中需要用到的参数及其默认值
        required_params = {
            'trailing_tp_increment_pct': '0.02',
            'trailing_sl_buffer_pct': '0.01',
            # 这里可以添加其他未来可能用到的参数
        }
        for key, default_value in required_params.items():
            # 优先使用数据库的值，否则使用默认值
            self.params[key] = all_params.get(key, Decimal(default_value))
        
        logger.info(f"策略参数加载完成: {self.params}")
    def _setup_backtest_schema(self, schema_name: str, initial_capital: Decimal):
        logger.info(f"--- 1. 在 Schema '{schema_name}' 中准备回测环境 ---")
        
        logger.info("正在新 Schema 中创建表结构 (执行 migrate)...")
        with connections['default'].cursor() as cursor:
            logger.info(f"临时隔离 search_path 到 '{schema_name}' 以便运行 migrate 命令。")
            cursor.execute(f'SET search_path TO "{schema_name}";')
            
            logger.info("正在新 Schema 中创建表结构 (执行 migrate)...")
            # 在这个隔离的环境下，migrate 看不到 public.django_migrations，因此会创建所有表。
            call_command('migrate')
            logger.info("表结构创建完成。")

        tables_to_copy = [
            'tb_stock_info', 'tb_daily_quotes', 'tb_corporate_actions',
            'tb_factor_definitions', 'tb_strategy_parameters', 
            'tb_daily_factor_values','tb_daily_trading_plan','tb_index_quotes_csi300'
        ]
        
        logger.info(f"准备从 'public' schema 复制基础数据到 '{schema_name}'...")
        with transaction.atomic(), connections['default'].cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema_name}";')
            for table_name in tables_to_copy:
                logger.info(f"  - 正在处理表: {table_name}")
                # 1. 区分并获取 "普通索引" 和 "约束"
                # =========================================================================
                # 1a. 获取普通索引 (不包括由 UNIQUE 或 PRIMARY KEY 约束创建的索引)
                logger.info(f"    - 正在获取 '{table_name}' 的普通索引...")
                cursor.execute("""
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = %s AND tablename = %s
                    AND indexname NOT IN (
                        SELECT conname FROM pg_constraint WHERE conrelid = %s::regclass
                    );
                """, [schema_name, table_name, f'"{schema_name}"."{table_name}"'])
                plain_indexes_to_recreate = [row[0] for row in cursor.fetchall()]
                # 1b. 获取约束 (外键和唯一约束)
                logger.info(f"    - 正在获取 '{table_name}' 的外键和唯一约束...")
                cursor.execute("""
                    SELECT 'ALTER TABLE ' || quote_ident(conrelid::regclass::text) || ' ADD CONSTRAINT ' || quote_ident(conname) || ' ' || pg_get_constraintdef(oid)
                    FROM pg_constraint
                    WHERE contype IN ('f', 'u') AND conrelid = %s::regclass;
                """, [f'"{schema_name}"."{table_name}"'])
                constraints_to_recreate = [row[0] for row in cursor.fetchall()]
                # 2. 删除索引和约束 (删除约束会自动删除其底层索引)
                # =========================================================================
                # 2a. 删除约束
                for const_def in constraints_to_recreate:
                    const_name = const_def.split('ADD CONSTRAINT ')[1].split(' ')[0]
                    logger.info(f"      - 删除约束: {const_name}")
                    cursor.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS {const_name};')
                
                # 2b. 删除普通索引
                for index_def in plain_indexes_to_recreate:
                    # 从 "CREATE INDEX index_name ON ..." 中提取 index_name
                    try:
                        index_name = index_def.split(' ')[2]
                        logger.info(f"      - 删除索引: {index_name}")
                        cursor.execute(f'DROP INDEX IF EXISTS "{index_name}";')
                    except IndexError:
                        logger.warning(f"无法从 '{index_def}' 解析索引名称，跳过删除。")
                # 3. 高效复制数据 (现在非常快)
                # =========================================================================
                logger.info(f"    - 正在从 public.{table_name} 复制数据...")
                sql = f'INSERT INTO "{table_name}" SELECT * FROM public."{table_name}";'
                cursor.execute(sql)
                logger.info(f"    - 数据复制完成。")
                # 4. 重建索引和约束
                # =========================================================================
                logger.info(f"    - 正在重建 '{table_name}' 的索引和约束...")
                # 4a. 重建普通索引
                for index_def in plain_indexes_to_recreate:
                    logger.info(f"      - 重建索引: {index_def}")
                    cursor.execute(index_def)
                
                # 4b. 重建约束 (这会自动重建它们的底层索引)
                for const_def in constraints_to_recreate:
                    logger.info(f"      - 重建约束: {const_def}")
                    cursor.execute(const_def)

                # =========================================================================
                # 5. 重置自增主键序列 (解决主键冲突的关键)
                # =========================================================================
                # 自动查找并更新当前表的自增序列
                find_serial_sql = """
                    SELECT 
                        a.attname, 
                        pg_get_serial_sequence(
                            quote_ident(n.nspname) || '.' || quote_ident(c.relname), 
                            a.attname
                        )
                    FROM 
                        pg_class c
                    JOIN 
                        pg_attribute a ON a.attrelid = c.oid
                    JOIN 
                        pg_namespace n ON c.relnamespace = n.oid -- 通过namespace OID关联
                    WHERE 
                        n.nspname = %s      -- 参数1: schema的名称
                        AND c.relname = %s  -- 参数2: 表的名称
                        AND a.attnum > 0 
                        AND NOT a.attisdropped
                        AND pg_get_serial_sequence(quote_ident(n.nspname) || '.' || quote_ident(c.relname), a.attname) IS NOT NULL;
                """
                cursor.execute(find_serial_sql, [schema_name, table_name])
                serial_columns = cursor.fetchall()

                for column_name, sequence_name in serial_columns:
                    logger.info(f"    - 发现自增列 '{column_name}'，正在重置其序列 '{sequence_name}'...")
                    
                    # 将序列的下一个值设置为 (表中该列的最大值 + 1)，如果表为空则设置为1
                    update_sequence_sql = f"""
                        SELECT setval(
                            '{sequence_name}', 
                            COALESCE((SELECT MAX("{column_name}") FROM "{table_name}"), 0) + 1, 
                            true
                        )
                    """
                    cursor.execute(update_sequence_sql)
                    logger.info(f"    - 序列 '{sequence_name}' 已更新。")

                
        logger.info("基础数据复制完成。")
        # with connections['default'].cursor() as cursor:
        #     for table_name in tables_to_copy:
        #         logger.info(f"  - 正在复制表: {table_name}")
        #         sql = f'INSERT INTO "{schema_name}"."{table_name}" SELECT * FROM public."{table_name}";'
        #         cursor.execute(sql)
        # logger.info("基础数据复制完成。")

        self.initial_capital = initial_capital
        self.cash_balance = self.initial_capital
        logger.info(f"初始资金已设定为: {self.initial_capital:.2f}")

    def _simulate_intraday_monitoring(self, position: Position, daily_quote: DailyQuotes):
        open_p, high_p, low_p = daily_quote.open, daily_quote.high, daily_quote.low
        
        # 临时变量，模拟盘中状态
        temp_position_state = {
            'current_stop_loss': position.current_stop_loss,
            'current_take_profit': position.current_take_profit
        }
        
        # 1. 开盘价检查
        decision = PositionMonitorLogic.check_and_decide(position, open_p, self.params)
        if decision['action'] == 'SELL':
            price=min(open_p,decision['exit_price'])
            logger.info(f"[回测] {position.stock_code_id} 开盘价 {open_p:.2f} 触发卖出，成交价 {decision['exit_price']:.2f}")
            self._persist_risk_prices_if_changed(
                position,
                temp_position_state['current_stop_loss'],
                temp_position_state['current_take_profit'],
                label='盘中'
            )
            logger.info(f"[回测] {position.stock_code_id} 开盘价 {open_p:.2f} 触发卖出，成交价 {price:.2f}")
            self.handler.sell_stock_by_market_price(position, decision['reason'], simulated_exit_price=price)
            return
        elif decision['action'] == 'UPDATE':
            temp_position_state.update(decision['updates'])
        # 2. 日内循环监控
        while True:
            action_taken_in_loop = False
            
            # 检查点1: 最低价是否触发卖出
            decision_low = PositionMonitorLogic.check_and_decide(
                Position( # 传入一个临时的、包含最新状态的Position对象
                    entry_price=position.entry_price,
                    current_stop_loss=temp_position_state['current_stop_loss'],
                    current_take_profit=temp_position_state['current_take_profit']
                ), 
                low_p, 
                self.params
            )
            if decision_low['action'] == 'SELL':
                self._persist_risk_prices_if_changed(
                    position,
                    temp_position_state['current_stop_loss'],
                    temp_position_state['current_take_profit'],
                    label='盘中'
                )
                logger.info(f"[回测] {position.stock_code_id} 最低价 {low_p:.2f} 触发卖出，成交价 {decision_low['exit_price']:.2f}")
                self.handler.sell_stock_by_market_price(position, decision_low['reason'], simulated_exit_price=decision_low['exit_price'])
                return
            # 检查点2: 最高价是否触发价格更新
            decision_high = PositionMonitorLogic.check_and_decide(
                Position(
                    entry_price=position.entry_price,
                    current_stop_loss=temp_position_state['current_stop_loss'],
                    current_take_profit=temp_position_state['current_take_profit']
                ), 
                high_p, 
                self.params
            )
            if decision_high['action'] == 'UPDATE':
                # 检查是否有实际的更新
                if temp_position_state != decision_high['updates']:
                    temp_position_state.update(decision_high['updates'])
                    action_taken_in_loop = True
            if not action_taken_in_loop:
                break
                
        # 3. 日终结算
        if temp_position_state['current_stop_loss'] != position.current_stop_loss or \
        temp_position_state['current_take_profit'] != position.current_take_profit:
            position.current_stop_loss = temp_position_state['current_stop_loss']
            position.current_take_profit = temp_position_state['current_take_profit']
            position.save(update_fields=['current_stop_loss', 'current_take_profit'])
            logger.info(f"[回测] {position.stock_code_id} 日终更新风控价格。SL: {position.current_stop_loss:.2f}, TP: {position.current_take_profit:.2f}")

    def run_backtest(self, start_date: str, end_date: str, initial_capital: Decimal) -> dict:
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        self.backtest_run_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"为本次回测创建临时 Schema: {self.backtest_run_id}")

        try:
            with connections['default'].cursor() as cursor:
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.backtest_run_id}";')

            with use_backtest_schema(self.backtest_run_id):
                self._setup_backtest_schema(self.backtest_run_id, initial_capital)
                self._load_parameters()
                handler = SimulateTradeHandler(self)
                self.handler=handler
                trading_days = self._get_trading_days()
                if not trading_days:
                    logger.error("在指定日期范围内未找到任何交易日，回测终止。")
                    return {}
                #master_panels = self._preload_data_for_backtest(trading_days)
                logger.info("--- [滚动窗口] 开始准备数据 ---")
                lookback_window_size = 250  # 因子计算所需的最大回溯期
                # 为了保证有足够的交易日，我们回溯更多的自然日
                extra_buffer_days = 20 
                
                # 1. 直接计算预加载的起始日期，不再扫描全表
                preload_start_date = trading_days[0] - timedelta(days=lookback_window_size + extra_buffer_days)
                
                logger.info(f"一次性查询数据库，时间窗口: {preload_start_date} to {self.end_date}")
                
                quotes_iterator = DailyQuotes.objects.filter(
                    trade_date__gte=preload_start_date,
                    trade_date__lte=self.end_date
                ).order_by('trade_date', 'stock_code_id').values(
                    'trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close'
                ).iterator(chunk_size=20000)
                logger.info("将迭代器数据按日期分组到字典中...")
                quotes_by_date = {}
                for row in quotes_iterator:
                    trade_date = row['trade_date']
                    if trade_date not in quotes_by_date:
                        quotes_by_date[trade_date] = []
                    quotes_by_date[trade_date].append(row)
                
                # 2. 初始化第一个滚动窗口
                logger.info("初始化第一个滚动窗口面板...")
                # 获取所有已加载的日期，并排序
                all_loaded_dates = sorted(quotes_by_date.keys())
                
                # 找到第一个回测日
                first_backtest_day = trading_days[0]
                
                # 从已加载的日期中，筛选出用于初始化的部分
                initial_window_dates = [d for d in all_loaded_dates if d <= first_backtest_day]
                initial_rows = []
                for d in initial_window_dates:
                    initial_rows.extend(quotes_by_date.get(d, []))
                
                if not initial_rows:
                    raise ValueError("初始化滚动窗口失败，没有获取到任何数据。")
                df_window = pd.DataFrame(initial_rows)
                df_window['trade_date'] = pd.to_datetime(df_window['trade_date'])
                
                rolling_panels = {}
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']:
                    panel = df_window.pivot(index='trade_date', columns='stock_code_id', values=col).astype(float)
                    rolling_panels[col] = panel
                
                logger.info(f"滚动窗口初始化完成，包含 {len(initial_window_dates)} 天数据。")
                logger.info(f"--- 2. 开始日度回测循环 ({len(trading_days)}天) ---")
                
                last_sent_month = None # 用于邮件触发

                for i, current_day in enumerate(trading_days):
                    self.current_date = current_day
                    logger.info(f"\n{'='*20} 模拟日: {self.current_date} ({i+1}/{len(trading_days)}) {'='*20}")
                    if i > 0:
                        # 检查新的一天是否有数据，有才滚动
                        new_day_data = quotes_by_date.get(current_day)
                        if new_day_data:
                            logger.debug(f"滚动窗口: 移除 {rolling_panels['close'].index[0].date()}, 添加 {current_day}")
                            # 移除最老的一天
                            for key in rolling_panels:
                                rolling_panels[key] = rolling_panels[key].iloc[1:]
                            
                            # 添加新的一天
                            df_new_day = pd.DataFrame(new_day_data)
                            df_new_day['trade_date'] = pd.to_datetime(df_new_day['trade_date'])
                            
                            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']:
                                new_row = df_new_day.pivot(index='trade_date', columns='stock_code_id', values=col).astype(float)
                                # 使用 pd.concat 合并，比 append 更推荐
                                rolling_panels[col] = pd.concat([rolling_panels[col], new_row])
                        else:
                            logger.warning(f"日期 {current_day} 在预加载数据中不存在，窗口未滚动。")

                    logger.info("-> [T日 盘前校准] ...")
                    before_fix_service = BeforeFixService(execution_date=self.current_date)
                    before_fix_service.run()
                    
                    self._handle_dividends()

                    logger.info("-> [T日 开盘决策与买入] ...")
                    decision_order_service = DecisionOrderService(handler=handler, execution_date=self.current_date)
                    decision_order_service.adjust_trading_plan_daily()
                    
                    while True:
                        open_positions_count = Position.objects.filter(status=Position.StatusChoices.OPEN).count()
                        max_pos = decision_order_service.current_max_positions
                        if open_positions_count >= max_pos: break
                        
                        self.last_buy_trade_id = None
                        decision_order_service.execute_orders()
                        
                        if self.last_buy_trade_id:
                            decision_order_service.calculate_stop_profit_loss(self.last_buy_trade_id)
                            # 在止盈止损计算完毕后，获取更新后的持仓对象
                            trade_log = TradeLog.objects.get(pk=self.last_buy_trade_id)
                            position = trade_log.position
                            # 现在才调用日志记录函数，此时 position 对象已包含正确的止盈止损价
                            handler._record_buy_operation(position)
                        else:
                            break

                    positions_to_monitor = Position.objects.filter(status=Position.StatusChoices.OPEN).exclude(
                        entry_datetime__date=self.current_date
                    )
                
                    for pos in positions_to_monitor:
                        # 如果持仓已在循环中被卖出，则跳过
                        if pos.status != Position.StatusChoices.OPEN:
                            continue
                        
                        try:
                            daily_quote = DailyQuotes.objects.get(stock_code_id=pos.stock_code_id, trade_date=self.current_date)
                            self._simulate_intraday_monitoring(pos, daily_quote)
                        except DailyQuotes.DoesNotExist:
                            logger.warning(f"[回测] 无法找到 {pos.stock_code_id} 在 {self.current_date} 的行情，当日无法监控。")
                            continue

                    


                    logger.info(f"-> [T日 盘后选股] 基于 {self.current_date} 的数据为下一交易日做准备...")
                    current_day_dt = pd.to_datetime(current_day)
                    selection_service = SelectionService(trade_date=self.current_date, mode='backtest',preloaded_panels=rolling_panels)
                    selection_service.run_selection()
                    
                    self._record_daily_log()

                    # --- 邮件发送逻辑 ---
                    is_last_day = (i == len(trading_days) - 1)
                    current_month = current_day.month
                    send_mail_flag = False

                    if is_last_day:
                        send_mail_flag = True
                        logger.info("回测结束，触发最终邮件报告。")
                    elif last_sent_month is not None and current_month != last_sent_month:
                        send_mail_flag = True
                        logger.info(f"月份从 {last_sent_month} 变为 {current_month}，触发月度邮件报告。")
                    
                    if send_mail_flag:
                        reporter = BacktestReporter(
                            schema_name=self.backtest_run_id,
                            start_date=self.start_date,
                            current_date=self.current_date,
                            initial_capital=self.initial_capital
                        )
                        reporter.send_report()
                    
                    last_sent_month = current_month
                    # --- 邮件发送逻辑结束 ---

                logger.info("--- 3. 回测循环结束 ---")
                return self._calculate_performance_metrics()

        except Exception as e:
            logger.critical(f"回测过程中发生严重错误: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_trading_days(self) -> list[date]:
        dates = DailyQuotes.objects.filter(
            trade_date__gte=self.start_date,
            trade_date__lte=self.end_date
        ).values_list('trade_date', flat=True).distinct().order_by('trade_date')
        return list(dates)

    def _handle_dividends(self):
        dividend_events = CorporateAction.objects.filter(
            ex_dividend_date=self.current_date, event_type=CorporateAction.EventType.DIVIDEND
        )
        if not dividend_events.exists(): return

        events_by_stock = {}
        for event in dividend_events:
            events_by_stock.setdefault(event.stock_code, []).append(event)
        
        open_positions = Position.objects.filter(
            stock_code_id__in=events_by_stock.keys(), status=Position.StatusChoices.OPEN
        )
        for pos in open_positions:
            for event in events_by_stock.get(pos.stock_code_id, []):
                dividend_amount = event.dividend_per_share * pos.quantity
                self.cash_balance += dividend_amount
                logger.info(f"除息事件: 持仓ID {pos.position_id} ({pos.stock_code_id}) 获得分红 {dividend_amount:.2f}")

    def _record_daily_log(self):
        open_positions = Position.objects.filter(status=Position.StatusChoices.OPEN)
        market_value = Decimal('0.0')
        for pos in open_positions:
            try:
                quote = DailyQuotes.objects.get(stock_code_id=pos.stock_code_id, trade_date=self.current_date)
                market_value += quote.close * pos.quantity
            except DailyQuotes.DoesNotExist:
                market_value += pos.entry_price * pos.quantity
        
        total_assets = self.cash_balance + market_value

        try:
            m_value_obj = DailyFactorValues.objects.get(
                stock_code_id=MARKET_INDICATOR_CODE,
                factor_code_id='dynamic_M_VALUE',
                trade_date=self.current_date
            )
            m_value = m_value_obj.raw_value
        except DailyFactorValues.DoesNotExist:
            m_value = None

        BacktestDailyLog.objects.create(
            backtest_run_id=self.backtest_run_id,
            trade_date=self.current_date,
            total_assets=total_assets,
            cash=self.cash_balance,
            holdings_value=market_value,
            market_m_value=m_value
        )
        logger.info(f"--- 日终结算 ({self.current_date}) ---")
        logger.info(f"现金: {self.cash_balance:.2f}, 持仓市值: {market_value:.2f}, 总资产: {total_assets:.2f}, M值: {m_value}")

    def _calculate_performance_metrics(self) -> dict:
        logger.info("--- 4. 计算回测性能指标 ---")
        daily_logs = BacktestDailyLog.objects.filter(backtest_run_id=self.backtest_run_id).order_by('trade_date')
        if not daily_logs.exists():
            return {}

        df = pd.DataFrame(list(daily_logs.values('total_assets')))
        df['total_assets'] = df['total_assets'].astype(float)
        
        final_value = df['total_assets'].iloc[-1]
        total_return_rate = (final_value / float(self.initial_capital)) - 1
        
        total_days = (self.end_date - self.start_date).days
        if total_days > 0:
            annualized_return = ((final_value / float(self.initial_capital)) ** (365.0 / total_days)) - 1
        else:
            annualized_return = 0.0

        df['peak'] = df['total_assets'].cummax()
        df['drawdown'] = (df['total_assets'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()

        result = {
            'total_return_rate': f"{total_return_rate:.2%}",
            'annualized_return': f"{annualized_return:.2%}",
            'max_drawdown': f"{max_drawdown:.2%}"
        }
        logger.info(f"最终回测结果: {result}")
        return result

