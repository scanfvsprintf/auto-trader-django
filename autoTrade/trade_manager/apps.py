# trade_manager/apps.py

import os
import sys
from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class TradeManagerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trade_manager'

    def ready(self):
        # # 关键修正：通过检查环境变量 RUN_MAIN 来防止调度器在重载主进程中启动两次
        # # 这个环境变量是 Django 的 autoreloader 在启动子进程时设置的。
        # # 我们只想在运行实际应用的子进程中启动调度器。
        # if os.environ.get('RUN_MAIN'):
        #     logger.info("检测到 Django 应用工作进程，准备初始化调度器...")
        #     from .service import scheduler_service
        #     # 确保调度器只启动一次
        #     if not scheduler_service.scheduler.running:
        #          scheduler_service.start()
        #     else:
        #          logger.warning("调度器已在运行，跳过重复启动。")
        # else:
        #     logger.info("检测到 Django 管理或重载主进程，跳过调度器初始化。")
        return

