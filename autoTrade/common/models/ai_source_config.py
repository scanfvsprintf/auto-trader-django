from django.db import models
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError


class AiSourceConfig(models.Model):
    """
    AI资源配置表
    用于存储各种AI服务提供商的配置信息
    """
    
    name = models.CharField(
        max_length=100, 
        unique=True, 
        verbose_name="资源名称",
        help_text="AI资源的名称，如OpenAI、Claude等"
    )
    
    url = models.URLField(
        max_length=500,
        verbose_name="资源URL",
        help_text="AI服务的API端点URL"
    )
    
    api_key = models.CharField(
        max_length=500,
        verbose_name="API密钥",
        help_text="访问AI服务的API密钥"
    )
    
    is_active = models.BooleanField(
        default=True,
        verbose_name="是否启用",
        help_text="是否启用此AI资源"
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
        help_text="AI资源的详细描述"
    )
    
    class Meta:
        db_table = 'tb_ai_source_config'
        verbose_name = "AI资源配置"
        verbose_name_plural = "AI资源配置"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({'启用' if self.is_active else '禁用'})"
    
    def clean(self):
        """验证模型数据"""
        super().clean()
        
        # 验证URL格式
        validator = URLValidator()
        try:
            validator(self.url)
        except ValidationError:
            raise ValidationError({'url': '请输入有效的URL格式'})
        
        # 验证API密钥格式
        if not self.api_key or len(self.api_key.strip()) < 10:
            raise ValidationError({'api_key': 'API密钥长度不能少于10个字符'})
    
    def save(self, *args, **kwargs):
        """保存前进行数据清理"""
        self.full_clean()
        super().save(*args, **kwargs)
