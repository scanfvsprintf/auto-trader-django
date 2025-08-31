# selection_manager/management/commands/generate_m_value_csv.py

import logging
import os
from decimal import Decimal

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from tqdm import tqdm

from common.models import IndexQuotesCsi300
from selection_manager.service.selection_service import SelectionService

# 获取logger实例
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    """
    一个Django管理命令，用于计算沪深300指数从2010-08-19至2025-08-18的
    完整历史M值，并生成一个CSV文件供分析脚本使用。
    
    运行方式:
    python manage.py generate_m_value_csv
    """
    help = '计算沪深300历史M值并生成 m_value_csi300.csv 文件'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('===== 开始计算沪深300历史M值 ====='))

        try:
            # 1. 一次性从数据库加载所有需要的沪深300行情数据
            self.stdout.write("正在从数据库加载沪深300历史行情数据...")
            quotes_qs = IndexQuotesCsi300.objects.all().order_by('trade_date').values('trade_date', 'close')
            if not quotes_qs:
                self.stdout.write(self.style.ERROR("错误：数据库表 tb_index_quotes_csi300 中没有数据。请先回填数据。"))
                return
            
            all_quotes_df = pd.DataFrame.from_records(quotes_qs)
            all_dates = all_quotes_df['trade_date'].tolist()
            self.stdout.write(f"数据加载完成，共 {len(all_dates)} 个交易日。")

            # 2. 初始化SelectionService并加载一次参数
            # 使用第一个日期进行初始化，后续在循环中更新日期
            service = SelectionService(trade_date=all_dates[0], mode='backtest')
            service._load_dynamic_parameters_and_defs()
            self.stdout.write("策略服务和参数初始化完成。")

            # 3. 循环计算每一天的M值
            results = []
            self.stdout.write("开始逐日计算M值...")
            LOOKBACK_BUFFER = 100
            if len(all_dates) <= LOOKBACK_BUFFER:
                self.stdout.write(self.style.ERROR(f"错误：总交易日数 ({len(all_dates)}) 不足以满足 {LOOKBACK_BUFFER} 天的回溯期。"))
                return
            # 使用tqdm创建进度条
            for current_date in tqdm(all_dates[LOOKBACK_BUFFER:], desc="计算M值进度"):
                # 更新服务的交易日期
                service.trade_date = current_date
                
                # 调用核心方法计算M值，按要求传入空stock_pool
                m_value = service._calculate_market_regime_M(stock_pool=[])
                
                # 从已加载的DataFrame中获取收盘价
                close_price = all_quotes_df.loc[all_quotes_df['trade_date'] == current_date, 'close'].iloc[0]
                
                results.append({
                    '日期': current_date,
                    'm值': m_value,
                    '沪深300收盘指数': float(close_price) # 转换为float以便pandas处理
                })

            if not results:
                self.stdout.write(self.style.WARNING("计算完成，但没有生成任何结果。"))
                return

            # 4. 将结果转换为DataFrame并保存为CSV
            self.stdout.write("所有M值计算完成，正在生成CSV文件...")
            results_df = pd.DataFrame(results)
            
            # 定义输出路径为项目根目录
            output_path = os.path.join(settings.BASE_DIR, 'm_value_csi300.csv')
            
            # 保存文件，使用 utf-8-sig 编码以确保Excel能正确打开
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.4f')

            self.stdout.write(self.style.SUCCESS(f"\n===== 任务成功完成！ ====="))
            self.stdout.write(f"文件已保存至: {output_path}")
            self.stdout.write(f"现在你可以运行 [M值关系分析.py] 脚本了。")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"任务执行过程中发生严重错误: {e}"))
            logger.error("生成M值CSV文件失败", exc_info=True)

