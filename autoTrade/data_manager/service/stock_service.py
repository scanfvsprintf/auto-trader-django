import logging
import datetime
from decimal import Decimal, ROUND_HALF_UP,InvalidOperation
import akshare as ak
import pandas as pd
from django.utils import timezone
from django.db import connection,transaction, DatabaseError

# 导入您的Django模型
from common.models.stock_info import StockInfo
from common.models.daily_quotes import DailyQuotes
from common.models.factor_definitions import FactorDefinitions
from common.models.daily_factor_values import DailyFactorValues
from common.models.strategy_parameters import StrategyParameters
from common.models.daily_trading_plan import DailyTradingPlan
from common.models.positions import Position
from common.models.trade_log import TradeLog
from common.models.system_log import SystemLog
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 获取logger实例
logger = logging.getLogger(__name__)

# 定义模块常量，便于维护
MODULE_NAME = 'data_manager'

class StockService:
    """
    封装了与股票数据相关的服务，包括从akshare更新数据和从本地数据库查询数据。
  
    使用示例 (在Django views.py 或 management command中):
  
    from .services.stock_service import StockService
  
    def my_view(request):
        service = StockService()
      
        # 示例1: 更新所有A股今天的行情
        service.update_local_a_shares()
      
        # 示例2: 更新指定几只股票某时间段的行情
        codes = ['sh.600519', 'sz.000001']
        service.update_local_a_shares(stock_codes=codes, start_date='2023-01-01', end_date='2023-01-31')
      
        # 示例3: 查询指定股票的基础信息
        stock_infos = service.query_stock_info(stock_codes=codes)
      
        # 示例4: 查询所有股票今天的日线行情
        daily_quotes = service.query_daily_quotes()
    """

    def _log_and_save(self, message: str, level: str = SystemLog.LogLevelChoices.INFO):
        """
        一个辅助方法，用于同时向标准logger和数据库系统日志表写入日志。
        """
        log_map = {
            SystemLog.LogLevelChoices.INFO: logger.info,
            SystemLog.LogLevelChoices.WARNING: logger.warning,
            SystemLog.LogLevelChoices.ERROR: logger.error,
            SystemLog.LogLevelChoices.CRITICAL: logger.critical,
        }
      
        # 打印到标准日志
        log_function = log_map.get(level, logger.info)
        log_function(message)
      
        # 保存到数据库
        # try:
        #     SystemLog.objects.create(
        #         log_level=level,
        #         module_name=MODULE_NAME,
        #         message=message
        #     )
        # except Exception as e:
        #     logger.error(f"无法将日志写入数据库: {e}")

    def _save_quotes_df_to_db(self, quotes_df: pd.DataFrame):
        """
        辅助方法：将一个DataFrame的行情数据通过 update_or_create 批量存入数据库。
        此方法具有幂等性，适用于所有数据，无需区分历史和当日。
        """
        if quotes_df.empty:
            return
 
        # 数据清洗和预处理
        quotes_df.fillna(0, inplace=True)
        quotes_df = quotes_df[(quotes_df['开盘'] > 0) & (quotes_df['收盘'] > 0) & (quotes_df['最高'] > 0) & (quotes_df['最低'] > 0) & (quotes_df['成交量'] >= 0)]
        if quotes_df.empty:
            self._log_and_save("数据清洗后，当前批次无有效数据可存储。", level=SystemLog.LogLevelChoices.INFO)
            return
            
        quotes_df['日期'] = pd.to_datetime(quotes_df['日期']).dt.date
        
        hfq_precision = Decimal('0.0000000001')
        records_to_process = len(quotes_df)
    
        try:
            # 将整个批次的 update_or_create 操作放在一个事务中，以提高性能
            with transaction.atomic():
                for _, row in quotes_df.iterrows():
                    try:
                        close_dec = Decimal(str(row['收盘']))
                        factor_dec = Decimal(str(row['复权因子']))
                        hfq_close_dec = (close_dec * factor_dec).quantize(hfq_precision, rounding=ROUND_HALF_UP)
                        
                        # 对每一行数据都执行 update_or_create
                        DailyQuotes.objects.update_or_create(
                            stock_code_id=row['stock_code'], 
                            trade_date=row['日期'],
                            defaults={
                                'open': Decimal(str(row['开盘'])), 
                                'high': Decimal(str(row['最高'])),
                                'low': Decimal(str(row['最低'])), 
                                'close': close_dec,
                                'volume': int(row['成交量']), 
                                'turnover': Decimal(str(row['成交额'])),
                                'adjust_factor': factor_dec, 
                                'hfq_close': hfq_close_dec
                            }
                        )
                    except (InvalidOperation, TypeError) as conversion_error:
                        self._log_and_save(f"跳过一条数据转换失败的记录: {row['stock_code']} on {row['日期']}. Error: {conversion_error}", level=SystemLog.LogLevelChoices.WARNING)
                        continue
            
            self._log_and_save(f"通过 update_or_create 成功处理了 {records_to_process} 条日线数据。")
    
        except (DatabaseError, Exception) as e:
            self._log_and_save(f"数据批量入库阶段(update_or_create)发生严重错误: {e}", level=SystemLog.LogLevelChoices.ERROR)

    def update_local_a_shares(
        self, 
        stock_codes: list[str] = None, 
        start_date: str = None, 
        end_date: str = None
    ):

        """
        1. 更新本地A股信息 (最终版：高效、健壮)
        """
        self._log_and_save(f"开始执行A股数据更新任务...")
        target_codes=[]
        # --- Part 1: 更新股票基础信息 (tb_stock_info) ---
        try:
            self._log_and_save("正在从交易所官方数据源获取全量A股列表...")
            
            # 1. 通过高效、可靠的接口一次性获取所有A股信息
            # 上海主板A股
            sh_main_df = ak.stock_info_sh_name_code(symbol="主板A股").copy()
            # 上海科创板
            sh_star_df = ak.stock_info_sh_name_code(symbol="科创板").copy()
            # 深圳A股
            sz_a_df = ak.stock_info_sz_name_code(symbol="A股列表").copy()
 
            # 2. 数据预处理和合并
            # 统一列名
            sh_main_df.rename(columns={'证券简称': 'stock_name', '上市日期': 'listing_date', '证券代码': 'code'}, inplace=True)
            sh_star_df.rename(columns={'证券简称': 'stock_name', '上市日期': 'listing_date', '证券代码': 'code'}, inplace=True)
            sz_a_df.rename(columns={'A股简称': 'stock_name', 'A股上市日期': 'listing_date', 'A股代码': 'code'}, inplace=True)
 
            # 添加市场前缀
            sh_main_df['code'] = 'sh.' + sh_main_df['code']
            sh_star_df['code'] = 'sh.' + sh_star_df['code']
            sz_a_df['code'] = 'sz.' + sz_a_df['code']
 
            # 合并为一个DataFrame
            all_stocks_df = pd.concat([
                sh_main_df[['code', 'stock_name', 'listing_date']],
                sh_star_df[['code', 'stock_name', 'listing_date']],
                sz_a_df[['code', 'stock_name', 'listing_date']]
            ], ignore_index=True)
 
            # 转换日期格式
            all_stocks_df['listing_date'] = pd.to_datetime(all_stocks_df['listing_date']).dt.date
            
            self._log_and_save(f"成功获取 {len(all_stocks_df)} 条A股基础信息。")
 
            # 3. 高效的批量入库操作
            with transaction.atomic():
                existing_stocks = StockInfo.objects.in_bulk(field_name='stock_code')
                
                to_create = []
                to_update = []
 
                for _, row in all_stocks_df.iterrows():
                    code = row['code']
                    stock_obj = existing_stocks.get(code)
                    
                    if not stock_obj:
                        # 如果股票不存在，则准备新建
                        to_create.append(
                            StockInfo(
                                stock_code=code,
                                stock_name=row['stock_name'],
                                listing_date=row['listing_date'],
                                status=StockInfo.StatusChoices.LISTING
                            )
                        )
                    elif stock_obj.stock_name != row['stock_name']:
                        # 如果股票存在但名称有变，则准备更新
                        stock_obj.stock_name = row['stock_name']
                        to_update.append(stock_obj)
 
                # 批量创建
                if to_create:
                    StockInfo.objects.bulk_create(to_create, batch_size=500)
                    self._log_and_save(f"批量新增 {len(to_create)} 条股票基础信息。")
                
                # 批量更新
                if to_update:
                    StockInfo.objects.bulk_update(to_update, ['stock_name'], batch_size=500)
                    self._log_and_save(f"批量更新 {len(to_update)} 条股票基础信息。")
 
            # 如果未指定 stock_codes，则使用获取到的所有代码进行下一步
            if not stock_codes or len(stock_codes)==0:
                stock_codes = all_stocks_df['code'].tolist()
            else:
                # 如果指定了，则只处理指定的代码
                stock_codes = [code for code in stock_codes if code in all_stocks_df['code'].values]
            target_codes = stock_codes if stock_codes else all_stocks_df['code'].tolist()
        except Exception as e:
            self._log_and_save(f"更新股票基础信息时发生严重错误: {e}", level=SystemLog.LogLevelChoices.ERROR)
            return

        # --- Part 2: 更新日线行情 (串行获取、内存汇总、批量入库) ---
        self._log_and_save(f"开始为 {len(target_codes)} 只股票串行获取日线行情...")
        today_str = datetime.date.today().strftime('%Y%m%d')
        start_date_str = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d') if start_date else today_str
        end_date_str = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d') if end_date else today_str
        # 定义批处理参数
        batch_size = 50  # 每批处理50只股票，可以根据你的机器内存调整
        batch_quotes_list = []
        # 改为串行循环
        for i, code in enumerate(target_codes):
            ak_code = code.split('.')[1]
            logger.info(f"进度: [{i+1}/{len(target_codes)}] 正在获取 {code}...")
            try:
                df_normal = ak.stock_zh_a_hist(symbol=ak_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="")
                time.sleep(1.6) # 增加礼貌性延时，降低被封风险
                df_hfq = ak.stock_zh_a_hist(symbol=ak_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="hfq")
                
                if df_normal.empty or df_hfq.empty:
                    continue
 
                df = pd.merge(df_normal, df_hfq[['日期', '收盘']], on='日期', suffixes=('', '_hfq'))
                df['复权因子'] = df.apply(lambda row: row['收盘_hfq'] / row['收盘'] if row['收盘'] and row['收盘'] != 0 else 0, axis=1)
                df['stock_code'] = code
                batch_quotes_list.append(df)
                
                time.sleep(1.4) # 增加礼貌性延时，降低被封风险
 
            except Exception as e:
                self._log_and_save(f"获取 {code} 日线行情失败: {e}", level=SystemLog.LogLevelChoices.WARNING)
                continue
 
        # 检查是否达到批处理大小，或者已经是最后一只股票
            if (i + 1) % batch_size == 0 or (i + 1) == len(target_codes):
                if not batch_quotes_list:
                    continue # 如果这个批次是空的，就跳过

                self._log_and_save(f"处理批次 {i//batch_size + 1}，包含 {len(batch_quotes_list)} 只股票...")
                
                # 1. 合并当前批次的数据
                batch_master_df = pd.concat(batch_quotes_list, ignore_index=True)
                
                # 2. 将这个批次的数据存入数据库
                self._save_quotes_df_to_db(batch_master_df)
                
                # 3. 清空批次列表，释放内存，为下一批做准备
                batch_quotes_list = []
                self._log_and_save(f"批次 {i//batch_size + 1} 处理完毕，内存已释放。")
 
        self._log_and_save("A股数据更新任务全部执行完毕。")

    def query_stock_info(self, stock_codes: list[str] = None) -> dict[str, StockInfo]:
        """
        2. 查询本地A股基础信息
        直接查询 tb_stock_info。
        """
        queryset = StockInfo.objects.all()
        if stock_codes:
            queryset = queryset.filter(stock_code__in=stock_codes)
      
        return {stock.stock_code: stock for stock in queryset}

    def query_daily_quotes(
        self, 
        stock_codes: list[str] = None, 
        start_date: str = None, 
        end_date: str = None
    ) -> dict[str, list[DailyQuotes]]:
        """
        3. 查询本地A股交易信息
        直接查询 tb_daily_quotes。
        """
        # 设置默认日期为今天
        today = datetime.date.today()
        start_date = start_date or today.strftime('%Y-%m-%d')
        end_date = end_date or today.strftime('%Y-%m-%d')

        # 使用 select_related 优化查询，一次性获取关联的 StockInfo 对象
        # 使用 order_by 确保数据按股票和日期排序，便于后续分组
        queryset = DailyQuotes.objects.select_related('stock_code').filter(
            trade_date__gte=start_date,
            trade_date__lte=end_date
        ).order_by('stock_code', 'trade_date')

        if stock_codes:
            queryset = queryset.filter(stock_code__in=stock_codes)
      
        # 构建输出字典
        result = {}
        for quote in queryset:
            # 使用 stock_code_id 避免再次访问数据库
            # setdefault 是构建这种分组字典的优雅方式
            result.setdefault(quote.stock_code_id, []).append(quote)
          
        return result

    #清空所有数据
    def clear_all_data(self):
        with connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM tb_daily_factor_values;")
            cursor.execute(f"DELETE FROM tb_daily_quotes;")
            cursor.execute(f"DELETE FROM tb_daily_trading_plan;")
            cursor.execute(f"DELETE FROM tb_factor_definitions;")
            cursor.execute(f"DELETE FROM tb_positions;")
            cursor.execute(f"DELETE FROM tb_stock_info;")
            cursor.execute(f"DELETE FROM tb_strategy_parameters;")
            cursor.execute(f"DELETE FROM tb_system_log;")
            cursor.execute(f"DELETE FROM tb_trade_log;")