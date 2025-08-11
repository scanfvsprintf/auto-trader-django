from django.db import models
from django.utils import timezone

class SystemLog(models.Model):
    """
    4.1. 系统日志表 (tb_system_log)
    说明: 记录系统运行过程中的关键信息、警告和错误，便于监控和调试。
    """
    class LogLevelChoices(models.TextChoices):
        INFO = 'INFO', 'INFO'
        WARNING = 'WARNING', 'WARNING'
        ERROR = 'ERROR', 'ERROR'
        CRITICAL = 'CRITICAL', 'CRITICAL'

    log_id = models.BigAutoField(
        primary_key=True, 
        help_text="日志唯一ID"
    )
    log_time = models.DateTimeField(
        default=timezone.now, 
        editable=False,
        help_text="日志记录时间"
    )
    log_level = models.CharField(
        max_length=10, 
        choices=LogLevelChoices.choices,
        help_text="日志级别。枚举: INFO, WARNING, ERROR, CRITICAL"
    )
    module_name = models.CharField(
        max_length=50, 
        blank=True, 
        null=True,
        help_text="产生日志的模块名, 如 '日终选股', '开盘决策'"
    )
    message = models.TextField(
        help_text="日志内容, 如 '无合适买点', '下单API请求失败'"
    )

    def __str__(self):
        return f"[{self.log_time.strftime('%Y-%m-%d %H:%M:%S')}] [{self.log_level}] {self.message[:80]}"

    class Meta:
        db_table = 'tb_system_log'
        verbose_name = '系统日志'
        verbose_name_plural = verbose_name
