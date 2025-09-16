import logging
import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import akshare as ak
import pandas as pd
from django.utils import timezone
from django.db import connection, transaction, DatabaseError

# 导入您的Django模型
from common.models.fund_info import FundInfo
from common.models.fund_daily_quotes import FundDailyQuotes
from common.models.system_log import SystemLog
import time

# 获取logger实例
logger = logging.getLogger(__name__)

# 定义模块常量，便于维护
MODULE_NAME = 'data_manager'

class FundService:
    """
    封装了与基金数据相关的服务，包括从akshare更新数据和从本地数据库查询数据。
  
    使用示例 (在Django views.py 或 management command中):
  
    from .services.fund_service import FundService
  
    def my_view(request):
        service = FundService()
      
        # 示例1: 更新所有场内ETF今天的行情
        service.update_local_etf_shares()
      
        # 示例2: 更新指定几只基金某时间段的行情
        codes = ['sh.510050', 'sz.159001']
        service.update_local_etf_shares(fund_codes=codes, start_date='2023-01-01', end_date='2023-01-31')
      
        # 示例3: 查询指定基金的基础信息
        fund_infos = service.query_fund_info(fund_codes=codes)
      
        # 示例4: 查询所有基金今天的日线行情
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
            
        # 统一日期格式处理 - 确保日期格式正确
        quotes_df['日期'] = pd.to_datetime(quotes_df['日期']).dt.date
        
        hfq_precision = Decimal('0.0000000001')
        records_to_process = len(quotes_df)
        success_count = 0
        error_count = 0
    
        try:
            # 将整个批次的 update_or_create 操作放在一个事务中，以提高性能
            with transaction.atomic():
                for _, row in quotes_df.iterrows():
                    try:
                        fund_code = row['fund_code']
                        close_dec = Decimal(str(row['收盘']))
                        factor_dec = Decimal(str(row['复权因子']))
                        hfq_close_dec = (close_dec * factor_dec).quantize(hfq_precision, rounding=ROUND_HALF_UP)
                        
                        # 对每一行数据都执行 update_or_create
                        FundDailyQuotes.objects.update_or_create(
                            fund_code_id=fund_code, 
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
                        success_count += 1
                        
                    except (InvalidOperation, TypeError) as conversion_error:
                        self._log_and_save(f"跳过一条数据转换失败的记录: {row['fund_code']} on {row['日期']}. Error: {conversion_error}", level=SystemLog.LogLevelChoices.WARNING)
                        error_count += 1
                        continue
                    except Exception as e:
                        self._log_and_save(f"跳过一条数据存储失败的记录: {row['fund_code']} on {row['日期']}. Error: {e}", level=SystemLog.LogLevelChoices.WARNING)
                        error_count += 1
                        continue
            
            self._log_and_save(f"通过 update_or_create 成功处理了 {success_count} 条基金日线数据，跳过 {error_count} 条错误记录。")
    
        except (DatabaseError, Exception) as e:
            self._log_and_save(f"数据批量入库阶段(update_or_create)发生严重错误: {e}", level=SystemLog.LogLevelChoices.ERROR)

    def update_local_etf_shares(
        self, 
        fund_codes: list[str] = None, 
        start_date: str = None, 
        end_date: str = None
    ):
        """
        1. 更新本地场内ETF信息 (参考A股逻辑)
        """
        self._log_and_save(f"开始执行场内ETF数据更新任务...")
        target_codes = []
        
        # --- Part 1: 更新基金基础信息 (tb_fund_info) ---
        try:
            self._log_and_save("正在从交易所官方数据源获取全量场内ETF列表...")
            
            # 1. 通过akshare获取场内ETF信息
            # 上海ETF
            sh_etf_df = ak.fund_etf_spot_em().copy()
            
            # 过滤出ETF数据并添加市场前缀
            sh_etf_df = sh_etf_df[sh_etf_df['代码'].str.startswith('51') | sh_etf_df['代码'].str.startswith('52')]
            sh_etf_df['code'] = 'sh.' + sh_etf_df['代码']
            sh_etf_df['fund_name'] = sh_etf_df['名称']
            sh_etf_df['listing_date'] = pd.to_datetime('2020-01-01').date()  # 默认上市日期，实际应该从数据源获取
            
            # 深圳ETF (如果有的话)
            sz_etf_df = ak.fund_etf_spot_em().copy()
            sz_etf_df = sz_etf_df[sz_etf_df['代码'].str.startswith('15') | sz_etf_df['代码'].str.startswith('16')]
            sz_etf_df['code'] = 'sz.' + sz_etf_df['代码']
            sz_etf_df['fund_name'] = sz_etf_df['名称']
            sz_etf_df['listing_date'] = pd.to_datetime('2020-01-01').date()
            
            # 合并为一个DataFrame
            all_funds_df = pd.concat([
                sh_etf_df[['code', 'fund_name', 'listing_date']],
                sz_etf_df[['code', 'fund_name', 'listing_date']]
            ], ignore_index=True)
            
            self._log_and_save(f"成功获取 {len(all_funds_df)} 条场内ETF基础信息。")
 
            # 3. 高效的批量入库操作
            with transaction.atomic():
                existing_funds = FundInfo.objects.in_bulk(field_name='fund_code')
                
                to_create = []
                to_update = []
 
                for _, row in all_funds_df.iterrows():
                    code = row['code']
                    fund_obj = existing_funds.get(code)
                    
                    if not fund_obj:
                        # 如果基金不存在，则准备新建
                        to_create.append(
                            FundInfo(
                                fund_code=code,
                                fund_name=row['fund_name'],
                                fund_type=FundInfo.FundTypeChoices.ETF,
                                listing_date=row['listing_date'],
                                status=FundInfo.StatusChoices.LISTING
                            )
                        )
                    elif fund_obj.fund_name != row['fund_name']:
                        # 如果基金存在但名称有变，则准备更新
                        fund_obj.fund_name = row['fund_name']
                        to_update.append(fund_obj)
 
                # 批量创建
                if to_create:
                    FundInfo.objects.bulk_create(to_create, batch_size=500)
                    self._log_and_save(f"批量新增 {len(to_create)} 条基金基础信息。")
                
                # 批量更新
                if to_update:
                    FundInfo.objects.bulk_update(to_update, ['fund_name'], batch_size=500)
                    self._log_and_save(f"批量更新 {len(to_update)} 条基金基础信息。")
 
            # 如果未指定 fund_codes，则使用获取到的所有代码进行下一步
            if not fund_codes or len(fund_codes) == 0:
                fund_codes = all_funds_df['code'].tolist()
            else:
                # 如果指定了，则只处理指定的代码
                fund_codes = [code for code in fund_codes if code in all_funds_df['code'].values]
            target_codes = fund_codes if fund_codes else all_funds_df['code'].tolist()
        except Exception as e:
            self._log_and_save(f"更新基金基础信息时发生严重错误: {e}", level=SystemLog.LogLevelChoices.ERROR)
            return

        # --- Part 2: 更新日线行情 (串行获取、内存汇总、批量入库) ---
        self._log_and_save(f"开始为 {len(target_codes)} 只基金串行获取日线行情...")
        today_str = datetime.date.today().strftime('%Y%m%d')
        start_date_str = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d') if start_date else today_str
        end_date_str = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d') if end_date else today_str
        
        self._log_and_save(f"日期范围: {start_date_str} 到 {end_date_str}")
        
        # 定义批处理参数
        batch_size = 50  # 每批处理50只基金，可以根据你的机器内存调整
        batch_quotes_list = []
        success_count = 0
        error_count = 0
        
        # 改为串行循环
        for i, code in enumerate(target_codes):
            # 正确提取akshare需要的纯数字代码
            if '.' in code:
                ak_code = code.split('.')[1]  # 从 'sh.510050' 提取 '510050'
            else:
                ak_code = code  # 如果已经是纯数字格式，直接使用
            
            logger.info(f"进度: [{i+1}/{len(target_codes)}] 正在获取 {code} (ak_code: {ak_code})...")
            try:
                # 使用基金历史数据接口
                df_normal = ak.fund_etf_hist_em(symbol=ak_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="")
                time.sleep(1.6) # 增加礼貌性延时，降低被封风险
                df_hfq = ak.fund_etf_hist_em(symbol=ak_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="hfq")
                
                if df_normal.empty or df_hfq.empty:
                    self._log_and_save(f"获取 {code} 数据为空，跳过", level=SystemLog.LogLevelChoices.WARNING)
                    error_count += 1
                    continue
 
                # 合并不复权和后复权数据
                df = pd.merge(df_normal, df_hfq[['日期', '收盘']], on='日期', suffixes=('', '_hfq'))
                df['复权因子'] = df.apply(lambda row: row['收盘_hfq'] / row['收盘'] if row['收盘'] and row['收盘'] != 0 else 0, axis=1)
                df['fund_code'] = code  # 使用完整的基金代码（如 'sh.510050'）
                
                # 验证数据完整性
                if len(df) == 0:
                    self._log_and_save(f"合并后 {code} 数据为空，跳过", level=SystemLog.LogLevelChoices.WARNING)
                    error_count += 1
                    continue
                
                batch_quotes_list.append(df)
                success_count += 1
                self._log_and_save(f"成功获取 {code} 数据，共 {len(df)} 条记录")
                
                time.sleep(1.4) # 增加礼貌性延时，降低被封风险
 
            except Exception as e:
                self._log_and_save(f"获取 {code} 日线行情失败: {e}", level=SystemLog.LogLevelChoices.WARNING)
                error_count += 1
                continue
 
            # 检查是否达到批处理大小，或者已经是最后一只基金
            if (i + 1) % batch_size == 0 or (i + 1) == len(target_codes):
                if not batch_quotes_list:
                    continue # 如果这个批次是空的，就跳过

                self._log_and_save(f"处理批次 {i//batch_size + 1}，包含 {len(batch_quotes_list)} 只基金...")
                
                # 1. 合并当前批次的数据
                batch_master_df = pd.concat(batch_quotes_list, ignore_index=True)
                
                # 2. 将这个批次的数据存入数据库
                self._save_quotes_df_to_db(batch_master_df)
                
                # 3. 清空批次列表，释放内存，为下一批做准备
                batch_quotes_list = []
                self._log_and_save(f"批次 {i//batch_size + 1} 处理完毕，内存已释放。")
 
        self._log_and_save(f"场内ETF数据更新任务全部执行完毕。成功处理 {success_count} 只基金，失败 {error_count} 只基金。")

    def query_fund_info(self, fund_codes: list[str] = None) -> dict[str, FundInfo]:
        """
        2. 查询本地基金基础信息
        直接查询 tb_fund_info。
        """
        queryset = FundInfo.objects.all()
        if fund_codes:
            queryset = queryset.filter(fund_code__in=fund_codes)
      
        return {fund.fund_code: fund for fund in queryset}

    def query_daily_quotes(
        self, 
        fund_codes: list[str] = None, 
        start_date: str = None, 
        end_date: str = None
    ) -> dict[str, list[FundDailyQuotes]]:
        """
        3. 查询本地基金交易信息
        直接查询 tb_fund_daily_quotes。
        """
        # 设置默认日期为今天
        today = datetime.date.today()
        start_date = start_date or today.strftime('%Y-%m-%d')
        end_date = end_date or today.strftime('%Y-%m-%d')

        # 使用 select_related 优化查询，一次性获取关联的 FundInfo 对象
        # 使用 order_by 确保数据按基金和日期排序，便于后续分组
        queryset = FundDailyQuotes.objects.select_related('fund_code').filter(
            trade_date__gte=start_date,
            trade_date__lte=end_date
        ).order_by('fund_code', 'trade_date')

        if fund_codes:
            queryset = queryset.filter(fund_code__in=fund_codes)
      
        # 构建输出字典
        result = {}
        for quote in queryset:
            # 使用 fund_code_id 避免再次访问数据库
            # setdefault 是构建这种分组字典的优雅方式
            result.setdefault(quote.fund_code_id, []).append(quote)
          
        return result
