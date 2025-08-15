# data_manager/management/commands/backfill_csi300_data.py
import logging
import time
from datetime import date, timedelta
import pandas as pd
import akshare as ak
from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction
from decimal import Decimal

from common.models.index_quotes_csi300 import IndexQuotesCsi300

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '一次性回填沪深300指数的历史日线数据。'

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            '--years',
            type=int,
            default=15,
            help='要回填的历史数据年数，默认为15年。'
        )

    def handle(self, *args, **options):
        years_to_backfill = options['years']
        end_date = date.today()
        start_date = end_date - timedelta(days=years_to_backfill * 365)

        self.stdout.write(self.style.SUCCESS(f"===== 开始回填沪深300指数历史数据 ====="))
        self.stdout.write(f"数据范围: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")

        try:
            # 1. 拉取数据
            self.stdout.write("正在从akshare获取数据...")
            df = ak.index_zh_a_hist(
                symbol="000300",
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d')
            )
            self.stdout.write(f"成功获取 {len(df)} 条数据。")

            if df.empty:
                self.stdout.write(self.style.WARNING("未获取到任何数据，任务终止。"))
                return

            # 2. 数据清洗和转换
            df.rename(columns={
                '日期': 'trade_date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                '成交量': 'volume', '成交额': 'amount', '振幅': 'amplitude',
                '涨跌幅': 'pct_change', '涨跌额': 'change_amount', '换手率': 'turnover_rate'
            }, inplace=True)
            
            # 成交量从“手”转换为“股”
            df['volume'] = df['volume'] * 100
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')


            # 3. 批量入库
            records_to_create = []
            for _, row in df.iterrows():
                records_to_create.append(
                    IndexQuotesCsi300(
                        trade_date=row['trade_date'],
                        open=Decimal(str(row['open'])),
                        close=Decimal(str(row['close'])),
                        high=Decimal(str(row['high'])),
                        low=Decimal(str(row['low'])),
                        volume=int(row['volume']),
                        amount=Decimal(str(row['amount'])),
                        amplitude=Decimal(str(row['amplitude'])),
                        pct_change=Decimal(str(row['pct_change'])),
                        change_amount=Decimal(str(row['change_amount'])),
                        turnover_rate=Decimal(str(row['turnover_rate'])) if pd.notna(row['turnover_rate']) else None
                    )
                )
            
            self.stdout.write("正在将数据写入数据库 (update_or_create)...")
            # with transaction.atomic():
            #     for record in records_to_create:
            #         IndexQuotesCsi300.objects.update_or_create(
            #             trade_date=record.trade_date,
            #             defaults={
            #                 'open': record.open, 'close': record.close, 'high': record.high, 'low': record.low,
            #                 'volume': record.volume, 'amount': record.amount, 'amplitude': record.amplitude,
            #                 'pct_change': record.pct_change, 'change_amount': record.change_amount,
            #                 'turnover_rate': record.turnover_rate
            #             }
            #         )
            # 【用这段代码替换上面删除的部分】
            self.stdout.write("正在将数据批量写入数据库...")
            # 定义需要更新的字段列表
            update_fields = [
                'open', 'close', 'high', 'low', 'volume', 'amount', 
                'amplitude', 'pct_change', 'change_amount', 'turnover_rate'
            ]
            # 使用 bulk_create 进行批量“更新或创建”
            IndexQuotesCsi300.objects.bulk_create(
                records_to_create,
                batch_size=10000,  # 推荐设置批次大小，防止内存占用过高
                update_conflicts=True,
                unique_fields=['trade_date'],  # 冲突判断的唯一键
                update_fields=update_fields  # 发生冲突时需要更新的字段
            )
            
            self.stdout.write(self.style.SUCCESS("===== 沪深300数据回填成功！ ====="))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"回填过程中发生错误: {e}"))
            logger.error("回填沪深300数据失败", exc_info=True)
