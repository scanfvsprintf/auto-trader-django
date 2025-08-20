# trade_manager/management/commands/run_m_distribution_backtest.py

from django.core.management.base import BaseCommand, CommandParser
from trade_manager.service.m_distribution_backtest_service import MDistributionBacktestService

class Command(BaseCommand):
    help = '运行一个新的、基于M值胜率分布的回测模块'

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            '--start',
            type=str,
            required=True,
            help='回测起始日期 (格式: YYYY-MM-DD)'
        )
        parser.add_argument(
            '--end',
            type=str,
            required=True,
            help='回测结束日期 (格式: YYYY-MM-DD)'
        )

        parser.add_argument(
            '--single_strategy_mode',
            action='store_true', # 这会创建一个布尔标志，如果命令行包含此参数，则其值为True
            help='开启单策略M值胜率分布回测模式，会额外为每个策略独立运行回测。'
        )

    def handle(self, *args, **options):
        start_date = options['start']
        end_date = options['end']
        single_strategy_mode = options['single_strategy_mode']
        self.stdout.write(self.style.SUCCESS('===== 开始执行M值胜率分布回测 ====='))
        self.stdout.write(f'  - 起始日期: {start_date}')
        self.stdout.write(f'  - 结束日期: {end_date}')

        try:
            service = MDistributionBacktestService(start_date=start_date, end_date=end_date,single_strategy_mode=single_strategy_mode)
            service.run()
            
            self.stdout.write(self.style.SUCCESS('\n===== M值胜率分布回测执行完毕 ====='))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'回测过程中发生严重错误: {e}'))
            raise e

