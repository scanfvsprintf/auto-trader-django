# ==============================================================================
# 文件 2/5: trade_manager/management/commands/run_backtest.py (新增)
# 描述: 用于从命令行启动回测的 Command 文件。
# ==============================================================================
from django.core.management.base import BaseCommand, CommandParser
from trade_manager.service.simulate_trade import SimulateTradeService
from decimal import Decimal

class Command(BaseCommand):
    help = '运行一个完整的交易策略回测'

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
            '--capital',
            type=Decimal,
            required=True,
            help='初始资金'
        )

    def handle(self, *args, **options):
        start_date = options['start']
        end_date = options['end']
        initial_capital = options['capital']

        self.stdout.write(self.style.SUCCESS(f'===== 开始执行回测任务 ====='))
        self.stdout.write(f'  - 起始日期: {start_date}')
        self.stdout.write(f'  - 结束日期: {end_date}')
        self.stdout.write(f'  - 初始资金: {initial_capital:.2f}')

        try:
            service = SimulateTradeService()
            # 注意：我们将所有参数都传递给 run_backtest 方法
            result = service.run_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            self.stdout.write(self.style.SUCCESS('\n===== 回测执行完毕 ====='))
            self.stdout.write(f'最终性能指标: {result}')

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'回测过程中发生严重错误: {e}'))
            # 在生产环境中，可能需要更详细的错误处理和日志记录
            raise e

