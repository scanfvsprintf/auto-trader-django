# data_manager/management/commands/migrate_to_pg.py

from django.core.management.base import BaseCommand
from data_manager.service.db_service import DbMigrationService

class Command(BaseCommand):
    help = '将整个SQLite数据库的结构和数据迁移到PostgreSQL'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('开始执行数据库迁移任务...'))
        
        try:
            service = DbMigrationService()
            service.migrate()
            self.stdout.write(self.style.SUCCESS('数据库迁移任务已成功完成。'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'数据库迁移失败: {e}'))

