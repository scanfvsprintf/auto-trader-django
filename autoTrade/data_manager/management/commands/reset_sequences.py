# data_manager/management/commands/reset_sequences.py (V2 - 修正版)
import logging
from django.core.management.base import BaseCommand
from django.db import connection, models
from django.apps import apps

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Resets PostgreSQL sequences for integer AutoFields to the max value of their primary key columns.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('===== 开始智能重置数据库序列 (仅处理自增整数主键) ====='))
        
        all_models = apps.get_models()
        
        with connection.cursor() as cursor:
            for model in all_models:
                table_name = model._meta.db_table
                pk_field = model._meta.pk
                
                # --- 核心修正：增加类型检查 ---
                # 1. 检查主键是否存在且是否为自增字段
                if not pk_field or not isinstance(pk_field, models.AutoField):
                    self.stdout.write(f'正在处理表: {table_name} ... ' + self.style.WARNING('跳过 (主键非自增整数)'))
                    continue

                # 2. 如果是自增字段，其内部类型一定是整数，可以安全处理
                pk_name = pk_field.name
                sequence_name = f"{table_name}_{pk_name}_seq"
                
                self.stdout.write(f'正在处理表: {table_name} (序列: {sequence_name}) ... ', ending='')
                
                try:
                    # SQL to get the max PK value.
                    # COALESCE is still useful for empty tables.
                    # The third argument 'false' in setval means the next value will be max_id + 1
                    # No change needed here as we've already filtered for integer PKs.
                    sql = f"""
                    SELECT setval('"{sequence_name}"', (SELECT COALESCE(MAX("{pk_name}"), 1) FROM "{table_name}"), false);
                    """
                    cursor.execute(sql)
                    self.stdout.write(self.style.SUCCESS('OK'))
                except Exception as e:
                    # 捕获其他可能的错误，例如序列真的不存在
                    if "does not exist" in str(e):
                        self.stdout.write(self.style.WARNING(f'跳过 (序列不存在)'))
                    else:
                        self.stdout.write(self.style.ERROR(f'失败: {e}'))
                        logger.error(f"重置序列 {sequence_name} 失败: {e}", exc_info=True)

        self.stdout.write(self.style.SUCCESS('\n===== 数据库序列智能重置完毕 ====='))
