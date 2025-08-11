# trade_manager/management/commands/run_scheduler.py

from django.core.management.base import BaseCommand
from trade_manager.service import scheduler_service

class Command(BaseCommand):
    help = '启动自动化交易的 APScheduler 调度器'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('正在启动调度器服务...'))
        scheduler_service.start_scheduler()
        self.stdout.write(self.style.SUCCESS('调度器服务已停止。'))

