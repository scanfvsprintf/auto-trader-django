# trade_manager/management/commands/run_scheduler.py

from django.core.management.base import BaseCommand
from trade_manager.service import scheduler_service
import logging
import time
logger = logging.getLogger(__name__)
class Command(BaseCommand):
    help = '启动自动化交易的 APScheduler 调度器'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('正在启动调度器服务...'))
        scheduler_service.start()
        try:
            # 这是关键：让主进程进入一个无限循环，以防止脚本退出
            # 这样后台的调度器线程才能一直存活
            while True:
                time.sleep(1)  # 每秒检查一次，降低CPU占用
        except (KeyboardInterrupt, SystemExit):
            # 当接收到退出信号时（如Ctrl+C或uWSGI的停止命令）
            # 优雅地关闭调度器
            logger.info("接收到退出信号，正在关闭调度器...")
            scheduler_service.scheduler.shutdown()
            logger.info("调度器已成功关闭。")
            self.stdout.write(self.style.SUCCESS('调度器服务已优雅地停止。'))
        self.stdout.write(self.style.SUCCESS('调度器服务已停止。'))

