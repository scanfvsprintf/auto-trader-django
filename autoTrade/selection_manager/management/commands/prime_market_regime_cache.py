# selection_manager/management/commands/prime_market_regime_cache.py

import logging
import time
from datetime import date, timedelta

import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from decimal import Decimal

from common.models import DailyQuotes, StockInfo, DailyFactorValues, StrategyParameters
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE

# 配置日志
logger = logging.getLogger('prime_market_regime_cache')

# 尝试导入GPU库
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("cuDF 和 cuPy 库已找到，将使用 GPU 进行计算。")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("未找到 cuDF 或 cuPy 库。此命令需要GPU环境。请安装相关依赖后重试。")
    logger.warning("参考安装: conda install -c rapidsai -c nvidia -c conda-forge cudf cupy")


class Command(BaseCommand):
    help = '使用GPU预热市场状态M(t)的历史数据缓存，用于首次运行或数据回补。'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=750,
            help='要预热的交易日天数，默认为750 (约3年)。'
        )
        parser.add_argument(
            '--end_date',
            type=str,
            # default=date.today().isoformat(),
            default="2023-04-30",
            help='预热的结束日期，默认为今天。格式: YYYY-MM-DD'
        )

    def handle(self, *args, **options):
        if not GPU_AVAILABLE:
            self.stdout.write(self.style.ERROR("GPU环境不可用，命令终止。"))
            return

        days_to_prime = options['days']
        end_date = date.fromisoformat(options['end_date'])
        
        self.stdout.write(self.style.SUCCESS(f"===== 开始M(t)缓存预热任务 (GPU模式) ====="))
        self.stdout.write(f"预热周期: {days_to_prime} 个交易日, 截止日期: {end_date}")
        
        total_start_time = time.time()

        # 1. 获取所需参数和交易日历
        params = {p.param_name: float(p.param_value) for p in StrategyParameters.objects.filter(param_name__startswith='dynamic_')}
        min_liquidity = params.get('dynamic_min_liquidity', 100000000)
        lookback_new_stock = int(params.get('dynamic_lookback_new_stock', 60))

        trade_dates = list(DailyQuotes.objects.filter(trade_date__lte=end_date)
                           .values_list('trade_date', flat=True).distinct().order_by('-trade_date')[:days_to_prime + 60])
        trade_dates.reverse()

        if not trade_dates:
            self.stdout.write(self.style.ERROR("数据库中无交易日数据，无法预热。"))
            return

        # 2. 加载全周期数据到Pandas
        self.stdout.write("正在从数据库加载全周期行情数据到内存...")
        start_load_time = time.time()
        all_quotes_qs = DailyQuotes.objects.filter(trade_date__in=trade_dates).values(
            'trade_date', 'stock_code_id', 'close', 'turnover', 'hfq_close'
        )
        all_stocks_qs = StockInfo.objects.values('stock_code', 'listing_date', 'stock_name')
        
        df_quotes = pd.DataFrame.from_records(all_quotes_qs)
        df_stocks = pd.DataFrame.from_records(all_stocks_qs)
        
        df_quotes['trade_date'] = pd.to_datetime(df_quotes['trade_date'])
        df_stocks['listing_date'] = pd.to_datetime(df_stocks['listing_date'])
        
        df = pd.merge(df_quotes, df_stocks, left_on='stock_code_id', right_on='stock_code')
        load_duration = time.time() - start_load_time
        self.stdout.write(f"数据加载完成，共 {len(df)} 条记录，耗时: {load_duration:.2f} 秒。")

        # 3. 数据传输到GPU
        self.stdout.write("正在将数据传输到GPU显存...")
        start_gpu_transfer_time = time.time()
        gdf = cudf.from_pandas(df)
        gpu_transfer_duration = time.time() - start_gpu_transfer_time
        self.stdout.write(f"数据成功传输到GPU，耗时: {gpu_transfer_duration:.2f} 秒。")

        # 4. 在GPU上进行计算
        self.stdout.write("正在GPU上并行计算所有日期的M(t)基础指标...")
        start_gpu_calc_time = time.time()
        
        # GPU计算逻辑
        results = []
        # 我们只对最近 `days_to_prime` 天进行计算和保存
        for calc_date in pd.to_datetime(trade_dates[-days_to_prime:]):
            # a. 当日筛选
            gdf_today = gdf[gdf['trade_date'] == calc_date]
            
            # 剔除ST
            gdf_today = gdf_today[~gdf_today['stock_name'].str.contains('ST')]
            
            # 剔除次新股
            min_listing_date = calc_date - timedelta(days=lookback_new_stock)
            gdf_today = gdf_today[gdf_today['listing_date'] < min_listing_date]
            
            # 剔除低流动性
            start_liquidity_date = calc_date - timedelta(days=40) # 多取一些数据
            gdf_liquidity_period = gdf[(gdf['trade_date'] >= start_liquidity_date) & (gdf['trade_date'] <= calc_date)]
            
            # 获取最近20个交易日
            recent_20_days = gdf_liquidity_period['trade_date'].unique().nlargest(20)
            gdf_liquidity_period = gdf_liquidity_period[gdf_liquidity_period['trade_date'].isin(recent_20_days)]
            
            avg_turnover = gdf_liquidity_period.groupby('stock_code_id')['turnover'].mean()
            liquid_stocks = avg_turnover[avg_turnover >= min_liquidity].index
            
            gdf_today = gdf_today[gdf_today['stock_code_id'].isin(liquid_stocks)]
            
            if gdf_today.empty:
                continue

            # b. 获取用于计算指标的历史窗口数据
            start_hist_date = calc_date - timedelta(days=120) # 넉넉하게 120일
            gdf_hist = gdf[(gdf['trade_date'] >= start_hist_date) & (gdf['trade_date'] <= calc_date)]
            gdf_hist = gdf_hist[gdf_hist['stock_code_id'].isin(gdf_today['stock_code_id'])]

            # c. 计算指标
            # M1: 创60日新高占比
            gdf_hist_60d = gdf_hist[gdf_hist['trade_date'].isin(gdf_hist['trade_date'].unique().nlargest(60))]
            high60 = gdf_hist_60d.groupby('stock_code_id')['close'].max()
            merged_m1 = gdf_today.merge(high60.rename('high60'), on='stock_code_id')
            m1 = (merged_m1['close'] >= merged_m1['high60']).sum() / len(merged_m1) if len(merged_m1) > 0 else 0

            # M2: MA60之上占比
            ma60 = gdf_hist_60d.groupby('stock_code_id')['close'].mean()
            merged_m2 = gdf_today.merge(ma60.rename('ma60'), on='stock_code_id')
            m2 = (merged_m2['close'] > merged_m2['ma60']).sum() / len(merged_m2) if len(merged_m2) > 0 else 0
            
            # M3: 60日回报率中位数
            date_t_minus_60 = gdf_hist_60d['trade_date'].unique().nsmallest(1).iloc[0]
            close_t = gdf_today[['stock_code_id', 'hfq_close']].set_index('stock_code_id')
            close_t_minus_60 = gdf_hist_60d[gdf_hist_60d['trade_date'] == date_t_minus_60][['stock_code_id', 'hfq_close']].set_index('stock_code_id')
            ret60 = (close_t / close_t_minus_60 - 1).dropna()
            m3 = ret60['hfq_close'].median() if not ret60.empty else 0

            # M4: 20日平均波动率
            gdf_hist_20d = gdf_hist[gdf_hist['trade_date'].isin(gdf_hist['trade_date'].unique().nlargest(20))]
            gdf_hist_20d = gdf_hist_20d.sort_values(by=['stock_code_id', 'trade_date'])
            returns = gdf_hist_20d.groupby('stock_code_id')['hfq_close'].pct_change().dropna()
            vol20 = gdf_hist_20d.merge(returns.rename('returns'), left_index=True, right_index=True).groupby('stock_code_id')['returns'].std()
            m4 = vol20.mean() if not vol20.empty else 0

            results.append({
                'trade_date': calc_date.date(),
                'dynamic_M1_RAW': float(m1),
                'dynamic_M2_RAW': float(m2),
                'dynamic_M3_RAW': float(m3) if m3 is not None else 0.0,
                'dynamic_M4_RAW': float(m4) if m4 is not None else 0.0,
            })
            self.stdout.write(f"  - 完成日期 {calc_date.date()} 的计算。")

        gpu_calc_duration = time.time() - start_gpu_calc_time
        self.stdout.write(f"GPU计算完成，耗时: {gpu_calc_duration:.2f} 秒。")

        # 5. 结果批量存入数据库
        self.stdout.write("正在将计算结果批量写入数据库缓存...")
        start_db_write_time = time.time()
        
        records_to_create = []
        for res in results:
            trade_date_res = res['trade_date']
            for factor_code_suffix, value in res.items():
                if factor_code_suffix == 'trade_date': continue
                if pd.notna(value):
                    records_to_create.append(DailyFactorValues(
                        stock_code_id=MARKET_INDICATOR_CODE,
                        trade_date=trade_date_res,
                        factor_code_id=factor_code_suffix,
                        raw_value=Decimal(str(value))
                    ))
        
        with transaction.atomic():
            # 先删除，再插入，保证幂等性
            dates_in_results = [r['trade_date'] for r in results]
            DailyFactorValues.objects.filter(
                stock_code_id=MARKET_INDICATOR_CODE,
                trade_date__in=dates_in_results
            ).delete()
            DailyFactorValues.objects.bulk_create(records_to_create, batch_size=500)

        db_write_duration = time.time() - start_db_write_time
        self.stdout.write(f"数据库写入完成，共 {len(records_to_create)} 条记录，耗时: {db_write_duration:.2f} 秒。")

        total_duration = time.time() - total_start_time
        self.stdout.write(self.style.SUCCESS(f"===== M(t)缓存预热任务成功完成！总耗时: {total_duration:.2f} 秒 ====="))

