# data_manager/service/db_service.py

import logging
import time
import pandas as pd
from sqlalchemy import create_engine, text
from django.apps import apps
from django.db import connections
from django.conf import settings
from collections import deque

logger = logging.getLogger(__name__)

class DbMigrationService:
    """
    一个健壮的服务，用于将数据从源数据库（SQLite）迁移到目标数据库（PostgreSQL）。
    它能自动处理表依赖关系，并使用分块读写来处理大数据表。
    """
    def __init__(self):
        # 从 Django settings 获取目标数据库配置
        pg_config = settings.DATABASES['default']
        self.pg_uri = f"postgresql+psycopg2://{pg_config['USER']}:{pg_config['PASSWORD']}@{pg_config['HOST']}:{pg_config['PORT']}/{pg_config['NAME']}"
        
        # 源数据库路径
        sqlite_path = settings.BASE_DIR / 'mainDB.sqlite3'
        self.sqlite_uri = f"sqlite:///{sqlite_path}"
        
        self.chunk_size = 50000  # 每次处理5万行，防止内存溢出

    def _get_migration_order(self) -> list:
        """
        通过拓扑排序分析 Django 模型之间的依赖关系，生成正确的迁移顺序。
        父表（被外键引用的表）会排在子表（有外键的表）前面。
        """
        all_models = apps.get_models()
        model_map = {model: model._meta.db_table for model in all_models}
        
        # 构建依赖图和入度计数
        dependencies = {model: set() for model in all_models}
        in_degree = {model: 0 for model in all_models}

        for model in all_models:
            for field in model._meta.get_fields():
                if field.is_relation and field.many_to_one and field.related_model in model_map:
                    # 如果 model 依赖于 related_model
                    related_model = field.related_model
                    if model in dependencies[related_model]:
                        continue
                    dependencies[related_model].add(model)
                    in_degree[model] += 1
        
        # 拓扑排序
        queue = deque([model for model in all_models if in_degree[model] == 0])
        sorted_models = []
        
        while queue:
            model = queue.popleft()
            sorted_models.append(model)
            
            for dependent_model in dependencies[model]:
                in_degree[dependent_model] -= 1
                if in_degree[dependent_model] == 0:
                    queue.append(dependent_model)

        if len(sorted_models) != len(all_models):
            raise Exception("数据库模型存在循环依赖，无法进行拓扑排序！")
            
        logger.info(f"计算出模型迁移顺序: {[model._meta.db_table for model in sorted_models]}")
        return sorted_models

    def migrate(self):
        """
        执行完整的数据库迁移流程。
        """
        logger.info("===== 开始数据库迁移：SQLite -> PostgreSQL =====")
        start_total_time = time.time()

        try:
            migration_order = self._get_migration_order()
            
            source_engine = create_engine(self.sqlite_uri)
            target_engine = create_engine(self.pg_uri)

            with target_engine.connect() as pg_conn:
                for model in migration_order:
                    if not(model._meta.db_table =='tb_daily_factor_values' or model._meta.db_table =='tb_daily_trading_plan' or model._meta.db_table =='tb_trade_log'):
                        logger.info(f"跳过表 {model}")
                        continue
                    else:
                        logger.info(f"执行表 {model}")
                    table_name = model._meta.db_table
                    logger.info(f"--- 正在迁移表: {table_name} ---")
                    start_table_time = time.time()

                    try:
                        # 1. 清空目标表并重置自增ID，保证幂等性
                        logger.info(f"清空目标表 {table_name}...")
                        # 使用 text() 来确保SQL语句被正确处理
                        truncate_sql = text(f'TRUNCATE TABLE public."{table_name}" RESTART IDENTITY CASCADE;')
                        pg_conn.execute(truncate_sql)
                        pg_conn.commit() # TRUNCATE 需要显式提交

                        # 2. 分块读取源数据并写入目标库
                        query = f'SELECT * FROM "{table_name}";'
                        total_rows = 0
                        for chunk_df in pd.read_sql_query(query, source_engine, chunksize=self.chunk_size):
                            
                            # 修正数据类型问题：Pandas有时会将bool转为int，需要转回来
                            for col in chunk_df.columns:
                                model_field = model._meta.get_field(col)
                                if model_field.get_internal_type() == 'BooleanField':
                                    chunk_df[col] = chunk_df[col].astype(bool)

                            chunk_df.to_sql(
                                name=table_name,
                                con=target_engine,
                                if_exists='append',
                                index=False,
                                method='multi',
                                schema='public' # 显式指定 schema
                            )
                            total_rows += len(chunk_df)
                            logger.info(f"已迁移 {total_rows} 行...")
                        
                        table_duration = time.time() - start_table_time
                        logger.info(f"表 {table_name} 迁移完成，共 {total_rows} 行，耗时 {table_duration:.2f} 秒。")

                    except Exception as e:
                        logger.error(f"迁移表 {table_name} 时发生错误: {e}", exc_info=True)
                        pg_conn.rollback() # 如果出错则回滚
                        raise  # 重新抛出异常，中断整个迁移过程

        except Exception as e:
            logger.critical(f"数据库迁移过程中发生严重错误，任务终止: {e}", exc_info=True)
            return

        total_duration = time.time() - start_total_time
        logger.info(f"===== 数据库迁移成功完成！总耗时: {total_duration:.2f} 秒 =====")

