from django.db import models
from django.core.exceptions import ValidationError
from .ai_source_config import AiSourceConfig


class AiModelConfig(models.Model):
    """
    AI模型配置表
    用于存储具体的AI模型配置信息
    """
    
    MODEL_TYPE_CHOICES = [
        (1, '文本模型'),
        (2, '生图模型'),
    ]
    
    name = models.CharField(
        max_length=100,
        unique=True,
        verbose_name="模型名称",
        help_text="AI模型的名称，如gpt-3.5-turbo、claude-3等"
    )
    
    model_type = models.IntegerField(
        choices=MODEL_TYPE_CHOICES,
        default=1,
        verbose_name="模型类型",
        help_text="模型类型：1-文本模型，2-生图模型"
    )
    
    source = models.ForeignKey(
        AiSourceConfig,
        on_delete=models.CASCADE,
        related_name='models',
        verbose_name="资源名称",
        help_text="关联的AI资源配置"
    )
    
    model_id = models.CharField(
        max_length=200,
        verbose_name="模型ID",
        help_text="在AI服务中的具体模型标识符"
    )
    
    is_active = models.BooleanField(
        default=True,
        verbose_name="是否启用",
        help_text="是否启用此AI模型"
    )
    
    max_tokens = models.IntegerField(
        default=1000,
        verbose_name="最大Token数",
        help_text="模型单次请求的最大Token数量"
    )
    
    temperature = models.FloatField(
        default=0.7,
        verbose_name="温度参数",
        help_text="控制模型输出的随机性，范围0-2"
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="创建时间"
    )
    
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name="更新时间"
    )
    
    description = models.TextField(
        blank=True,
        null=True,
        verbose_name="描述",
        help_text="AI模型的详细描述"
    )
    
    class Meta:
        db_table = 'tb_ai_model_config'
        verbose_name = "AI模型配置"
        verbose_name_plural = "AI模型配置"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"
    
    def clean(self):
        """验证模型数据"""
        super().clean()
        
        # 验证温度参数范围
        if not (0 <= self.temperature <= 2):
            raise ValidationError({'temperature': '温度参数必须在0-2之间'})
        
        # 验证最大Token数
        if self.max_tokens <= 0:
            raise ValidationError({'max_tokens': '最大Token数必须大于0'})
        
        # 验证模型ID
        if not self.model_id or len(self.model_id.strip()) < 1:
            raise ValidationError({'model_id': '模型ID不能为空'})
    
    def save(self, *args, **kwargs):
        """保存前进行数据清理"""
        self.full_clean()
        super().save(*args, **kwargs)
    
    @property
    def source_name(self):
        """获取关联的源名称"""
        return self.source.name if self.source else None
