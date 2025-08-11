import logging
import time
from datetime import datetime

import akshare
import pandas as pd
from django.db import transaction

# 导入您的 Django models
# 请根据您的项目结构调整以下导入路径
from common.models.corporate_action import CorporateAction
from common.models.stock_info import StockInfo

# 配置日志记录器
logger = logging.getLogger(__name__)
class CorporateActionService:
    def _fetch_and_save_split_events(self,stock_codes_filter: list, start_date: str, end_date: str):
        """
        预留的拆股/并股事件处理函数。
        """
        # logger.info(f"正在检查拆股/并股事件 (当前版本暂未实现)...")
        pass

    def sync_corporate_actions(self,start_date: str, end_date: str, stock_codes: list = None):
        """
        从 Akshare 高效同步指定日期范围和股票范围的股权事件数据，并存入数据库。
        """
        logger.info(f"开始同步股权事件，日期范围: {start_date} to {end_date}。")
        if stock_codes:
            logger.info(f"目标股票: {len(stock_codes)} 只。")
        else:
            logger.info("目标股票: 全部A股。")

        # 1. 任务开始前，一次性清理数据
        try:
            with transaction.atomic():
                qs = CorporateAction.objects.filter(
                    ex_dividend_date__gte=start_date,
                    ex_dividend_date__lte=end_date
                )
                if stock_codes:
                    qs = qs.filter(stock_code__in=stock_codes)
                
                deleted_count, _ = qs.delete()
                logger.info(f"数据清理完成。在 {start_date} 到 {end_date} 范围内共删除 {deleted_count} 条旧记录。")
        except Exception as e:
            logger.error(f"清理历史数据时发生严重错误，任务终止: {e}", exc_info=True)
            return

        all_stocks_map = {s.split('.')[-1]: s for s in StockInfo.objects.values_list('stock_code', flat=True)}
        ak_codes_filter = [c.split('.')[-1] for c in stock_codes] if stock_codes else None

        # 2. 处理分红、送股、转股 (stock_fhps_em)
        try:
            logger.info("开始处理分红、送股、转股事件...")
            fhps_dfs = []
            start_year = datetime.strptime(start_date, '%Y-%m-%d').year
            end_year = datetime.strptime(end_date, '%Y-%m-%d').year
            
            # ★★★★★ 优化点：使用更精确的年份范围，覆盖跨年预案 ★★★★★
            report_suffixes = ["0331", "0630", "0930", "1231"]
            for year in range(start_year - 1, end_year + 1):
                for suffix in report_suffixes:
                    report_date = f"{year}{suffix}"
                    logger.info(f"正在拉取报告期 {report_date} 的分红送配预案...")
                    try:
                        time.sleep(1)
                        fhps_df = akshare.stock_fhps_em(date=report_date)
                        if not fhps_df.empty:
                            fhps_dfs.append(fhps_df)
                    except Exception as e:
                        logger.warning(f"拉取报告期 {report_date} 数据失败或无数据: {e}")
            
            if fhps_dfs:
                # 使用 '代码' 和 '除权除息日' 作为联合主键去重，防止同一事件因在不同报告期披露而重复
                all_fhps_df = pd.concat(fhps_dfs, ignore_index=True).drop_duplicates(subset=['代码', '除权除息日'])
                
                all_fhps_df['除权除息日'] = pd.to_datetime(all_fhps_df['除权除息日'], errors='coerce')
                all_fhps_df.dropna(subset=['除权除息日'], inplace=True)
                
                mask = (all_fhps_df['除权除息日'] >= pd.to_datetime(start_date)) & (all_fhps_df['除权除息日'] <= pd.to_datetime(end_date))
                filtered_fhps_df = all_fhps_df[mask].copy()

                if ak_codes_filter:
                    filtered_fhps_df = filtered_fhps_df[filtered_fhps_df['代码'].isin(ak_codes_filter)]

                logger.info(f"共获取到 {len(filtered_fhps_df)} 条符合条件的分红送转记录，准备入库...")

                with transaction.atomic():
                    for _, row in filtered_fhps_df.iterrows():
                        ak_code = row['代码']
                        stock_code_prefixed = all_stocks_map.get(ak_code)
                        if not stock_code_prefixed:
                            continue

                        # 分红
                        if pd.notna(row['现金分红-现金分红比例']) and row['现金分红-现金分红比例'] > 0:
                            CorporateAction.objects.create(
                                stock_code=stock_code_prefixed,
                                ex_dividend_date=row['除权除息日'].date(),
                                record_date=pd.to_datetime(row['股权登记日'], errors='coerce').date() if pd.notna(row['股权登记日']) else None,
                                notice_date=pd.to_datetime(row['最新公告日期'], errors='coerce').date() if pd.notna(row['最新公告日期']) else None,
                                event_type=CorporateAction.EventType.DIVIDEND,
                                dividend_per_share=row['现金分红-现金分红比例'] / 10
                            )

                        # 送股
                        if pd.notna(row['送转股份-送转比例']) and row['送转股份-送转比例'] > 0:
                            CorporateAction.objects.create(
                                stock_code=stock_code_prefixed,
                                ex_dividend_date=row['除权除息日'].date(),
                                record_date=pd.to_datetime(row['股权登记日'], errors='coerce').date() if pd.notna(row['股权登记日']) else None,
                                notice_date=pd.to_datetime(row['最新公告日期'], errors='coerce').date() if pd.notna(row['最新公告日期']) else None,
                                event_type=CorporateAction.EventType.BONUS,
                                shares_before=10,
                                shares_after=10 + row['送转股份-送转比例']
                            )

                        # 转股
                        if pd.notna(row['送转股份-转股比例']) and row['送转股份-转股比例'] > 0:
                            CorporateAction.objects.create(
                                stock_code=stock_code_prefixed,
                                ex_dividend_date=row['除权除息日'].date(),
                                record_date=pd.to_datetime(row['股权登记日'], errors='coerce').date() if pd.notna(row['股权登记日']) else None,
                                notice_date=pd.to_datetime(row['最新公告日期'], errors='coerce').date() if pd.notna(row['最新公告日期']) else None,
                                event_type=CorporateAction.EventType.TRANSFER,
                                shares_before=10,
                                shares_after=10 + row['送转股份-转股比例']
                            )
            logger.info("分红、送股、转股事件处理完成。")
        except Exception as e:
            logger.error(f"处理分红送转数据时发生严重错误: {e}", exc_info=True)

        # 3. 处理配股 (stock_pg_em)
        try:
            logger.info("开始处理配股事件...")
            time.sleep(1)
            all_pg_df = akshare.stock_pg_em()
            
            # Akshare 返回的 '股权登记日' 可能包含无效日期，需要处理
            all_pg_df['股权登记日'] = pd.to_datetime(all_pg_df['股权登记日'], errors='coerce')
            all_pg_df.dropna(subset=['股权登记日'], inplace=True)
 
            mask = (all_pg_df['股权登记日'] >= pd.to_datetime(start_date)) & (all_pg_df['股权登记日'] <= pd.to_datetime(end_date))
            filtered_pg_df = all_pg_df[mask].copy()
 
            if ak_codes_filter:
                filtered_pg_df = filtered_pg_df[filtered_pg_df['股票代码'].isin(ak_codes_filter)]
            
            logger.info(f"共获取到 {len(filtered_pg_df)} 条符合条件的配股记录，准备入库...")
 
            with transaction.atomic():
                for _, row in filtered_pg_df.iterrows():
                    ak_code = row['股票代码']
                    stock_code_prefixed = all_stocks_map.get(ak_code)
                    if not stock_code_prefixed:
                        continue
 
                    # --- 修改开始 ---
                    # 从 '10配3.0' 这样的字符串中解析出配股比例数值
                    rights_ratio_val = 0
                    rights_ratio_str = row['配股比例']
                    
                    # 确保 '配股比例' 是一个有效的、可解析的字符串
                    if pd.notna(rights_ratio_str) and isinstance(rights_ratio_str, str) and '配' in rights_ratio_str:
                        try:
                            # 按 '配' 分割，取后面的部分，并转换为浮点数
                            ratio_str_part = rights_ratio_str.split('配')[1]
                            rights_ratio_val = float(ratio_str_part)
                        except (IndexError, ValueError) as e:
                            logger.warning(f"无法解析股票 {ak_code} 的配股比例 '{rights_ratio_str}'，已跳过。错误: {e}")
                            continue # 跳过此条记录
 
                    if rights_ratio_val > 0:
                        CorporateAction.objects.create(
                            stock_code=stock_code_prefixed,
                            # 注意：配股通常使用 '股权登记日' 作为关键日期，'除权日' 在此接口中可能不提供
                            ex_dividend_date=row['股权登记日'].date(), 
                            record_date=row['股权登记日'].date(),
                            notice_date=None, # akshare.stock_pg_em() 未提供公告日期
                            event_type=CorporateAction.EventType.RIGHTS,
                            shares_before=10, # 配股基准通常是10股
                            shares_after=10 + rights_ratio_val, # 使用解析后的数值
                            rights_issue_price=row['配股价']
                        )
                    # --- 修改结束 ---
 
            logger.info("配股事件处理完成。")
        except KeyError as e:
            # 捕获 '配股比例' 等字段不存在的错误
            logger.error(f"处理配股数据时发生字段缺失错误: {e}。请检查 Akshare 返回的数据列名是否已变更。", exc_info=True)
        except Exception as e:
            logger.error(f"处理配股数据时发生严重错误: {e}", exc_info=True)

        # 4. 调用预留的拆股/并股处理函数
        self._fetch_and_save_split_events(stock_codes, start_date, end_date)

        logger.info("所有股权事件同步任务已全部完成。")
