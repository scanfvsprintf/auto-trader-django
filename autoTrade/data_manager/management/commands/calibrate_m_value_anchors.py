# data_manager/management/commands/calibrate_m_value_anchors.py
import logging
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction
from decimal import Decimal

from common.models import IndexQuotesCsi300, StrategyParameters

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '一次性校准M值计算所需的固定分位锚点参数。'

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            '--start-date',
            type=str,
            default=None,  # 或者设置为 None
            help='用于校准的起始日期 (格式: YYYY-MM-DD)。'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            default=None,  # 或者设置为 None
            help='用于校准的截止日期 (格式: YYYY-MM-DD)。'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== 开始校准M值固定分位锚点 ====="))
        
        # 1. 加载数据
        # df = pd.DataFrame.from_records(IndexQuotesCsi300.objects.all().values())
        start_date = options['start_date']
        end_date = options['end_date']
        queryset = IndexQuotesCsi300.objects.all()
        if start_date:
            queryset = queryset.filter(trade_date__gte=start_date)
        if end_date:
            queryset = queryset.filter(trade_date__lte=end_date)
        # 可以在这里加一句日志，方便确认
        self.stdout.write(f"使用数据范围: 从 {start_date or '最早'} 到 {end_date or '最新'}")
        df = pd.DataFrame.from_records(queryset.values())
        if df.empty:
            self.stdout.write(self.style.ERROR("数据库中没有沪深300指数数据，请先执行 backfill_csi300_data。"))
            return
        
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        columns_to_convert = ['open', 'high', 'low', 'close', 'turnover_rate']
        for col in columns_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # 2. 计算四个基础指标的完整历史序列
        self.stdout.write("正在计算基础指标的历史序列...")
        df['ma60'] = df['close'].rolling(60).mean()
        df['m1_trend'] = (df['close'] - df['ma60']) / df['ma60']
        df['m2_momentum'] = df['close'].pct_change(20)
        df['daily_return'] = df['close'].pct_change()
        df['m3_volatility'] = df['daily_return'].rolling(20).std()
        avg_turnover_20 = df['turnover_rate'].rolling(20).mean()
        avg_turnover_60 = df['turnover_rate'].rolling(60).mean()
        df['m4_turnover'] = avg_turnover_20 / avg_turnover_60
        
        indicators_df = df[['m1_trend', 'm2_momentum', 'm3_volatility', 'm4_turnover']].dropna()
        self.stdout.write(f"指标历史计算完成，有效数据点: {len(indicators_df)}。")

        # 3. 计算分位锚点并准备参数
        params_to_update = {}
        quantiles = [0.10, 0.50, 0.90]
        indicator_map = {
            'm1_trend': 'trend',
            'm2_momentum': 'momentum',
            'm3_volatility': 'volatility',
            'm4_turnover': 'turnover'
        }

        for col, name in indicator_map.items():
            percentiles = indicators_df[col].quantile(quantiles)
            for q in quantiles:
                param_name = f"dynamic_m_csi300_anchor_{name}_p{int(q*100)}"
                param_value = Decimal(str(percentiles[q]))
                params_to_update[param_name] = param_value
                self.stdout.write(f"计算出锚点: {param_name} = {param_value:.4f}")

        # 4. 写入数据库
        self.stdout.write("正在将锚点参数写入数据库...")
        with transaction.atomic():
            for name, value in params_to_update.items():
                StrategyParameters.objects.update_or_create(
                    param_name=name,
                    defaults={
                        'param_value': value,
                        'group_name': 'M_CSI300_ANCHORS',
                        'description': f'沪深300的M值计算-指标{name.split("_p")[0].split("_")[-1]}-锚点P{name.split("_p")[-1]}'
                    }
                )
        
        self.stdout.write(self.style.SUCCESS(f"===== 成功更新 {len(params_to_update)} 个锚点参数！ ====="))
