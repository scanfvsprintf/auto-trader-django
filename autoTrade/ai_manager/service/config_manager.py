import logging
from typing import Dict, Any, List, Optional
from django.core.exceptions import ValidationError
from common.models import AiSourceConfig, AiModelConfig

logger = logging.getLogger(__name__)


class AiConfigManager:
    """
    AI配置管理工具类
    提供AI资源和模型配置的增删查改功能
    """
    
    # ==================== AI源配置管理 ====================
    
    @staticmethod
    def create_source_config(
        name: str,
        url: str,
        api_key: str,
        description: str = None,
        is_active: bool = True
    ) -> AiSourceConfig:
        """
        创建AI源配置
        
        Args:
            name: 资源名称
            url: 资源URL
            api_key: API密钥
            description: 描述
            is_active: 是否启用
            
        Returns:
            AiSourceConfig: 创建的配置对象
            
        Raises:
            ValidationError: 数据验证失败
        """
        try:
            source_config = AiSourceConfig(
                name=name,
                url=url,
                api_key=api_key,
                description=description,
                is_active=is_active
            )
            source_config.full_clean()
            source_config.save()
            
            logger.info(f"成功创建AI源配置: {name}")
            return source_config
            
        except Exception as e:
            logger.error(f"创建AI源配置失败: {str(e)}")
            raise ValidationError(f"创建AI源配置失败: {str(e)}")
    
    @staticmethod
    def get_source_config(source_id: int = None, name: str = None) -> Optional[AiSourceConfig]:
        """
        获取AI源配置
        
        Args:
            source_id: 源ID
            name: 源名称
            
        Returns:
            AiSourceConfig: 配置对象，如果不存在返回None
        """
        try:
            if source_id:
                return AiSourceConfig.objects.get(id=source_id)
            elif name:
                return AiSourceConfig.objects.get(name=name)
            else:
                return None
        except AiSourceConfig.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"获取AI源配置失败: {str(e)}")
            return None
    
    @staticmethod
    def list_source_configs(active_only: bool = False) -> List[AiSourceConfig]:
        """
        列出所有AI源配置
        
        Args:
            active_only: 是否只返回启用的配置
            
        Returns:
            List[AiSourceConfig]: 配置列表
        """
        try:
            queryset = AiSourceConfig.objects.all()
            if active_only:
                queryset = queryset.filter(is_active=True)
            return list(queryset.order_by('-created_at'))
        except Exception as e:
            logger.error(f"列出AI源配置失败: {str(e)}")
            return []
    
    @staticmethod
    def update_source_config(
        source_id: int,
        name: str = None,
        url: str = None,
        api_key: str = None,
        description: str = None,
        is_active: bool = None
    ) -> Optional[AiSourceConfig]:
        """
        更新AI源配置
        
        Args:
            source_id: 源ID
            name: 资源名称
            url: 资源URL
            api_key: API密钥
            description: 描述
            is_active: 是否启用
            
        Returns:
            AiSourceConfig: 更新后的配置对象，如果不存在返回None
        """
        try:
            source_config = AiSourceConfig.objects.get(id=source_id)
            
            if name is not None:
                source_config.name = name
            if url is not None:
                source_config.url = url
            if api_key is not None:
                source_config.api_key = api_key
            if description is not None:
                source_config.description = description
            if is_active is not None:
                source_config.is_active = is_active
            
            source_config.full_clean()
            source_config.save()
            
            logger.info(f"成功更新AI源配置: {source_config.name}")
            return source_config
            
        except AiSourceConfig.DoesNotExist:
            logger.warning(f"AI源配置不存在: {source_id}")
            return None
        except Exception as e:
            logger.error(f"更新AI源配置失败: {str(e)}")
            raise ValidationError(f"更新AI源配置失败: {str(e)}")
    
    @staticmethod
    def delete_source_config(source_id: int) -> bool:
        """
        删除AI源配置
        
        Args:
            source_id: 源ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            source_config = AiSourceConfig.objects.get(id=source_id)
            
            # 检查是否有关联的模型配置
            related_models = AiModelConfig.objects.filter(source=source_config)
            if related_models.exists():
                raise ValidationError(f"无法删除AI源配置，存在关联的模型配置: {related_models.count()}个")
            
            source_config.delete()
            logger.info(f"成功删除AI源配置: {source_config.name}")
            return True
            
        except AiSourceConfig.DoesNotExist:
            logger.warning(f"AI源配置不存在: {source_id}")
            return False
        except Exception as e:
            logger.error(f"删除AI源配置失败: {str(e)}")
            raise ValidationError(f"删除AI源配置失败: {str(e)}")
    
    # ==================== AI模型配置管理 ====================
    
    @staticmethod
    def create_model_config(
        name: str,
        model_type: int,
        source_id: int,
        model_id: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        description: str = None,
        is_active: bool = True
    ) -> AiModelConfig:
        """
        创建AI模型配置
        
        Args:
            name: 模型名称
            model_type: 模型类型（1-文本模型，2-生图模型）
            source_id: 关联的源ID
            model_id: 模型ID
            max_tokens: 最大Token数
            temperature: 温度参数
            description: 描述
            is_active: 是否启用
            
        Returns:
            AiModelConfig: 创建的配置对象
            
        Raises:
            ValidationError: 数据验证失败
        """
        try:
            # 验证源配置是否存在
            source_config = AiSourceConfig.objects.get(id=source_id)
            if not source_config.is_active:
                raise ValidationError("关联的AI源配置已禁用")
            
            model_config = AiModelConfig(
                name=name,
                model_type=model_type,
                source=source_config,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                description=description,
                is_active=is_active
            )
            model_config.full_clean()
            model_config.save()
            
            logger.info(f"成功创建AI模型配置: {name}")
            return model_config
            
        except AiSourceConfig.DoesNotExist:
            raise ValidationError(f"AI源配置不存在: {source_id}")
        except Exception as e:
            logger.error(f"创建AI模型配置失败: {str(e)}")
            raise ValidationError(f"创建AI模型配置失败: {str(e)}")
    
    @staticmethod
    def get_model_config(model_id: int = None, name: str = None) -> Optional[AiModelConfig]:
        """
        获取AI模型配置
        
        Args:
            model_id: 模型ID
            name: 模型名称
            
        Returns:
            AiModelConfig: 配置对象，如果不存在返回None
        """
        try:
            if model_id:
                return AiModelConfig.objects.get(id=model_id)
            elif name:
                return AiModelConfig.objects.get(name=name)
            else:
                return None
        except AiModelConfig.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"获取AI模型配置失败: {str(e)}")
            return None
    
    @staticmethod
    def list_model_configs(active_only: bool = False, model_type: int = None) -> List[AiModelConfig]:
        """
        列出所有AI模型配置
        
        Args:
            active_only: 是否只返回启用的配置
            model_type: 模型类型过滤
            
        Returns:
            List[AiModelConfig]: 配置列表
        """
        try:
            queryset = AiModelConfig.objects.select_related('source')
            if active_only:
                queryset = queryset.filter(is_active=True)
            if model_type is not None:
                queryset = queryset.filter(model_type=model_type)
            return list(queryset.order_by('-created_at'))
        except Exception as e:
            logger.error(f"列出AI模型配置失败: {str(e)}")
            return []
    
    @staticmethod
    def update_model_config(
        model_id: int,
        name: str = None,
        model_type: int = None,
        source_id: int = None,
        model_id_str: str = None,
        max_tokens: int = None,
        temperature: float = None,
        description: str = None,
        is_active: bool = None
    ) -> Optional[AiModelConfig]:
        """
        更新AI模型配置
        
        Args:
            model_id: 模型ID
            name: 模型名称
            model_type: 模型类型
            source_id: 关联的源ID
            model_id_str: 模型ID字符串
            max_tokens: 最大Token数
            temperature: 温度参数
            description: 描述
            is_active: 是否启用
            
        Returns:
            AiModelConfig: 更新后的配置对象，如果不存在返回None
        """
        try:
            model_config = AiModelConfig.objects.get(id=model_id)
            
            if name is not None:
                model_config.name = name
            if model_type is not None:
                model_config.model_type = model_type
            if source_id is not None:
                source_config = AiSourceConfig.objects.get(id=source_id)
                model_config.source = source_config
            if model_id_str is not None:
                model_config.model_id = model_id_str
            if max_tokens is not None:
                model_config.max_tokens = max_tokens
            if temperature is not None:
                model_config.temperature = temperature
            if description is not None:
                model_config.description = description
            if is_active is not None:
                model_config.is_active = is_active
            
            model_config.full_clean()
            model_config.save()
            
            logger.info(f"成功更新AI模型配置: {model_config.name}")
            return model_config
            
        except AiModelConfig.DoesNotExist:
            logger.warning(f"AI模型配置不存在: {model_id}")
            return None
        except Exception as e:
            logger.error(f"更新AI模型配置失败: {str(e)}")
            raise ValidationError(f"更新AI模型配置失败: {str(e)}")
    
    @staticmethod
    def delete_model_config(model_id: int) -> bool:
        """
        删除AI模型配置
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            model_config = AiModelConfig.objects.get(id=model_id)
            model_config.delete()
            logger.info(f"成功删除AI模型配置: {model_config.name}")
            return True
            
        except AiModelConfig.DoesNotExist:
            logger.warning(f"AI模型配置不存在: {model_id}")
            return False
        except Exception as e:
            logger.error(f"删除AI模型配置失败: {str(e)}")
            raise ValidationError(f"删除AI模型配置失败: {str(e)}")
    
    # ==================== 便捷方法 ====================
    
    @staticmethod
    def get_active_sources() -> List[Dict[str, Any]]:
        """获取所有启用的AI源配置"""
        sources = AiConfigManager.list_source_configs(active_only=True)
        return [
            {
                'id': source.id,
                'name': source.name,
                'url': source.url,
                'api_key': source.api_key,
                'description': source.description,
                'is_active': source.is_active,
                'created_at': source.created_at.isoformat(),
                'updated_at': source.updated_at.isoformat()
            }
            for source in sources
        ]
    
    @staticmethod
    def get_active_models(model_type: int = 1) -> List[Dict[str, Any]]:
        """获取所有启用的AI模型配置"""
        models = AiConfigManager.list_model_configs(active_only=True, model_type=model_type)
        return [
            {
                'id': model.id,
                'name': model.name,
                'model_type': model.model_type,
                'source_id': model.source.id,
                'source_name': model.source.name,
                'model_id': model.model_id,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'description': model.description,
                'is_active': model.is_active,
                'created_at': model.created_at.isoformat(),
                'updated_at': model.updated_at.isoformat()
            }
            for model in models
        ]
