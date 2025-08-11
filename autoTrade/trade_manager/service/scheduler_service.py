# trade_manager/service/scheduler_service.py

import logging
import pandas as pd # 修正：导入pandas
from datetime import date, timedelta, datetime

import akshare as ak
from apscheduler.schedulers.background import BackgroundScheduler # 使用BackgroundScheduler
from django.conf import settings
from django.db import transaction
from decimal import Decimal

from selection_manager.service.selection_service import SelectionService
from data_manager.service.corporate_action_service import CorporateActionService
from data_manager.service.stock_service import StockService
from data_manager.service.email_service import EmailNotificationService
from trade_manager.service.before_fix_service import BeforeFixService
from trade_manager.service.decision_order_service import DecisionOrderService
from trade_manager.service.monitor_exit_service import MonitorExitService
from trade_manager.service.real_trade_handler import RealTradeHandler, connection_manager
from common.models import TradeLog, Position
from common.config_loader import config_loader

logger = logging.getLogger(__name__)

class TradingCalendar:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TradingCalendar, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.trade_dates = set()
            self.last_updated = None
            self.initialized = True
            self._update_calendar()

    def _update_calendar(self):
        logger.info("正在更新交易日历...")
        try:
            df = ak.tool_trade_date_hist_sina()
            self.trade_dates = set(pd.to_datetime(df['trade_date']).dt.date)
            self.last_updated = date.today()
            logger.info(f"交易日历更新成功，共获取 {len(self.trade_dates)} 个交易日。")
        except Exception as e:
            logger.error(f"更新交易日历失败: {e}", exc_info=True)

    def is_trading_day(self, check_date: date) -> bool:
        if date.today() != self.last_updated:
            self._update_calendar()
        return check_date in self.trade_dates

trading_calendar = TradingCalendar()

# --- Job Functions ---

def run_job_wrapper(job_func, job_name, *args, **kwargs):
    scheduler_status = config_loader.get('scheduler', {}).get('status')
    if scheduler_status == 'off': return

    logger.info(f"--- [{job_name}] 任务触发 ---")
    if scheduler_status == 'dry_run':
        logger.info(f"[{job_name}] 空转模式，任务仅打印日志，不执行。")
        return
    
    try:
        job_func(*args, **kwargs)
        logger.info(f"--- [{job_name}] 任务成功执行 ---")
    except Exception as e:
        logger.error(f"--- [{job_name}] 任务执行失败: {e} ---", exc_info=True)

def daily_check():
    today = date.today()
    if not trading_calendar.is_trading_day(today):
        logger.debug(f"{today} 不是交易日，今日主要交易流程任务将跳过。")
        return False
    return True

def selection_job():
    

    t_minus_1 = date.today() - timedelta(days=1)
    if not trading_calendar.is_trading_day(date.today()):
        logger.info(f"今日({date.today()})不是交易日，不执行选股任务。")
        return
    service = StockService()
    service.update_local_a_shares(start_date=date.today().strftime('%Y-%m-%d'),end_date=date.today().strftime('%Y-%m-%d'))
    service = SelectionService(trade_date=date.today(), mode='realtime')
    service.run_selection()

def premarket_fix_job():
    if not daily_check(): return
    service = BeforeFixService(execution_date=date.today())
    service.run()

def opening_decision_job():
    if not daily_check(): return
    handler = RealTradeHandler()
    service = DecisionOrderService(handler, execution_date=date.today())
    
    logger.info("执行交易预案二次筛选...")
    service.adjust_trading_plan_daily()
    
    logger.info("循环执行下单，尝试填满仓位...")
    max_positions = service.current_max_positions
    logger.info(f"根据M(t)计算，当日动态最大持仓数为: {max_positions}")

    
    open_positions_count = Position.objects.filter(status=Position.StatusChoices.OPEN).count()
    slots_to_fill = max_positions - open_positions_count
 
    # 3. 循环调用同一个实例的方法
    for i in range(slots_to_fill):
        logger.info(f"尝试填充第 {i+1}/{slots_to_fill} 个仓位...")
        service.execute_orders()

def monitoring_job():
    if not daily_check(): return
    handler = RealTradeHandler()
    service = MonitorExitService(handler, execution_date=date.today())
    service.monitor_and_exit_positions()

def update_order_status_job():
    if not daily_check(): return
    handler = RealTradeHandler()
    if handler.is_simulation: return

    pending_trades = TradeLog.objects.filter(status=TradeLog.StatusChoices.PENDING, external_order_id__isnull=False)
    if not pending_trades.exists(): return
    
    try:
        real_orders = handler._api_get_orders()
        if not real_orders: return
        real_orders_map = {str(o['entrust_no']): o for o in real_orders}

        for trade in pending_trades:
            real_order = real_orders_map.get(trade.external_order_id)
            if not real_order: continue
            
            if real_order['order_status'] in ['已成', '全部成交']:
                with transaction.atomic():
                    trade.status = TradeLog.StatusChoices.FILLED
                    trade.price = Decimal(str(real_order['filled_price']))
                    # 注意：easytrader返回的佣金可能不准确，这里仅为示例
                    trade.commission = Decimal(str(real_order.get('business_balance', '0.0'))) - Decimal(str(real_order.get('clear_balance', '0.0')))
                    trade.save()

                    if trade.trade_type == 'buy':
                        decision_service = DecisionOrderService(handler, execution_date=date.today())
                        decision_service.calculate_stop_profit_loss(trade.trade_id)
                    else: # sell
                        position = trade.position
                        position.status = Position.StatusChoices.CLOSED
                        position.save()
                logger.info(f"订单 {trade.trade_id} (委托号: {trade.external_order_id}) 状态更新为已成交。")

            elif real_order['order_status'] in ['已撤', '废单', '部成已撤']:
                with transaction.atomic():
                    trade.status = TradeLog.StatusChoices.CANCELLED if '撤' in real_order['order_status'] else TradeLog.StatusChoices.FAILED
                    trade.save()
                    if trade.trade_type == 'buy':
                        position = trade.position
                        position.status = Position.StatusChoices.CLOSED
                        position.save()
                logger.info(f"订单 {trade.trade_id} (委托号: {trade.external_order_id}) 状态更新为 {trade.status}。")

    except Exception as e:
        logger.error(f"更新订单状态时出错: {e}", exc_info=True)

def update_corporate_actions_job():
    today = date.today()
    start_date = today - timedelta(days=30)
    end_date = today + timedelta(days=30)
    service = CorporateActionService()
    service.sync_corporate_actions(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

def disconnect_job():
    logger.info("执行每日断开连接任务...")
    connection_manager.disconnect()

scheduler = BackgroundScheduler(timezone='Asia/Shanghai')


def schedule_intraday_jobs():
    """在每个交易日开盘前，添加当天的盘中监控任务。"""
    job_id_monitor = 'intraday_monitoring_job'
    job_id_order_status = 'intraday_order_status_job'
    
    # 为防止重复添加，先尝试移除旧的（如果存在）
    try:
        scheduler.remove_job(job_id_monitor)
        logger.info(f"成功移除旧的盘中监控任务 (ID: {job_id_monitor})。")
    except Exception:
        pass # JobNotFoundError, a normal case
    
    try:
        scheduler.remove_job(job_id_order_status)
        logger.info(f"成功移除旧的订单状态更新任务 (ID: {job_id_order_status})。")
    except Exception:
        pass
 
    if not daily_check(): return
 
    today_str = date.today().isoformat()
    logger.info(f"正在为 {today_str} 添加盘中任务...")
 
    scheduler.add_job(
        run_job_wrapper, 
        'interval', 
        seconds=5, 
        start_date=f'{today_str} 09:30:01', 
        end_date=f'{today_str} 14:57:00', 
        args=[monitoring_job, '盘中监控'],
        id=job_id_monitor, # **给任务一个唯一的ID**
        replace_existing=True # 如果ID已存在，则替换
    )
 
    scheduler.add_job(
        run_job_wrapper, 
        'interval', 
        seconds=10, 
        start_date=f'{today_str} 09:30:00', 
        end_date=f'{today_str} 15:00:00', 
        args=[update_order_status_job, '更新订单状态'],
        id=job_id_order_status, # **给任务一个唯一的ID**
        replace_existing=True
    )
    logger.info("当日盘中任务已成功调度。")
 
 
# 清理任务的函数，虽然 replace_existing=True也能工作，但显式清理更干净
def cleanup_intraday_jobs():
    """收盘后清理，以防万一。"""
    try:
        scheduler.remove_job('intraday_monitoring_job')
        scheduler.remove_job('intraday_order_status_job')
        logger.info("已清理当日盘中任务。")
    except Exception:
        pass

# 邮件发送任务
def email_jobs():
    """每天发送计划邮件"""
    today = date.today()
    service = EmailNotificationService(today)
    service.runEmailSend()


def start():
    """启动调度器的主函数"""
    if config_loader.get('scheduler', {}).get('status') == 'off':
        logger.info("调度器状态为 'off'，不启动。")
        return

    if scheduler.running:
        logger.warning("调度器已在运行中。")
        return

    # 添加任务
    scheduler.add_job(run_job_wrapper, 'cron', day='*', hour=22, minute=0, args=[selection_job, '日终选股'])
    scheduler.add_job(run_job_wrapper, 'cron', day='*', hour=6, minute=30, args=[premarket_fix_job, '盘前校准'])
    scheduler.add_job(run_job_wrapper, 'cron', day='*', hour=7, minute=0, args=[email_jobs, '预案推送'])
    #scheduler.add_job(run_job_wrapper, 'cron', day='*', hour=9, minute=25, second=5, args=[opening_decision_job, '开盘决策'])
    
    # --- 每日动态任务的调度器 ---
    # 在每个交易日的开盘前（例如9:00）安排好当天的盘中任务
    #scheduler.add_job(schedule_intraday_jobs, 'cron', day='*', hour=9, minute=0)
    #在收盘后清理
    #scheduler.add_job(cleanup_intraday_jobs, 'cron', day='*', hour=15, minute=5)
    
    # 数据和连接管理任务
    scheduler.add_job(run_job_wrapper, 'cron', day='*', hour=6, minute=0, args=[update_corporate_actions_job, '更新除权除息'])
    #scheduler.add_job(run_job_wrapper, 'cron', day='*', hour=15, minute=30, args=[disconnect_job, '断开连接'])

    logger.info("APScheduler 已配置完成，准备在后台启动...")
    scheduler.start()
