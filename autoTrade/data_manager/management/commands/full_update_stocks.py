# data_manager/management/commands/full_update_stocks.py

from django.core.management.base import BaseCommand
from data_manager.service.stock_service import StockService
from selection_manager.service.selection_service import SelectionService
import time
from datetime import date,datetime
class Command(BaseCommand):
    help = '清空并重新获取过去五年的全部A股数据'

    def handle(self, *args, **options):
        total_start_time = time.time()
        self.stdout.write(self.style.SUCCESS('===== 开始执行全量数据更新任务 ====='))
        
        service = StockService()
        
        # 1. 清空所有旧数据
        self.stdout.write('正在清空所有历史数据...')
        #service.clear_all_data()
        self.stdout.write(self.style.SUCCESS('历史数据已清空。'))
        
        # 2. 按年份顺序获取数据
        #service.clear_all_data()
        #service.update_local_a_shares(start_date="2025-08-06",end_date="2025-08-08")
        service.update_local_a_shares(start_date="2015-01-01",end_date="2017-12-31")
        # service.update_local_a_shares(start_date="2023-01-01",end_date="2023-12-31")
        # service.update_local_a_shares(start_date="2022-01-01",end_date="2022-12-31")
        # service.update_local_a_shares(start_date="2021-01-01",end_date="2021-12-31")
        total_end_time = time.time()
        self.stdout.write(self.style.SUCCESS(f'\n===== 所有年份数据更新完毕！总耗时: {(total_end_time - total_start_time) / 3600:.2f} 小时 ====='))
        #self.stdout.write('开始预热M值...')
        #service=SelectionService(datetime.strptime('2025-08-08', "%Y-%m-%d").date())
        #service.run_selection()
        total_end_time_2 = time.time()
