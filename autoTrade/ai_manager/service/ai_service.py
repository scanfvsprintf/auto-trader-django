import requests
import json
import logging
from typing import Dict, Any, Optional, List
from django.conf import settings
from common.models import AiModelConfig, AiSourceConfig

logger = logging.getLogger(__name__)


class AiServiceException(Exception):
    """AI服务异常类"""
    pass


class AiService:
    """
    AI服务接口类
    提供标准化的AI模型调用接口
    """
    
    def __init__(self, model_name: str = None):
        """
        初始化AI服务
        
        Args:
            model_name: 模型名称，如果为None则使用默认模型
        """
        self.model_name = model_name
        self.model_config = None
        self.source_config = None
        self._load_model_config()
    
    def _load_model_config(self):
        """加载模型配置"""
        try:
            if self.model_name:
                self.model_config = AiModelConfig.objects.get(
                    name=self.model_name, 
                    is_active=True
                )
            else:
                # 使用默认的第一个活跃模型
                self.model_config = AiModelConfig.objects.filter(
                    is_active=True,
                    model_type=1  # 文本模型
                ).first()
            
            if not self.model_config:
                raise AiServiceException("未找到可用的AI模型配置")
            
            self.source_config = self.model_config.source
            if not self.source_config or not self.source_config.is_active:
                raise AiServiceException("AI源配置无效或已禁用")
                
        except AiModelConfig.DoesNotExist:
            raise AiServiceException(f"未找到模型配置: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型配置失败: {str(e)}")
            raise AiServiceException(f"加载模型配置失败: {str(e)}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        生成文本内容
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数，如temperature, max_tokens等
            
        Returns:
            str: 生成的文本内容
            
        Raises:
            AiServiceException: AI服务调用异常
        """
        try:
            # 构建请求参数
            request_data = self._build_request_data(prompt, **kwargs)
            
            # 发送请求
            response = self._send_request(request_data)
            
            # 解析响应
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"AI文本生成失败: {str(e)}")
            raise AiServiceException(f"AI文本生成失败: {str(e)}")
    
    def _build_request_data(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """构建请求数据"""
        # 使用模型配置的默认参数，允许kwargs覆盖
        temperature = kwargs.get('temperature', self.model_config.temperature)
        max_tokens = kwargs.get('max_tokens', self.model_config.max_tokens)
        
        # 根据不同的AI服务提供商构建不同的请求格式
        if 'openai' in self.source_config.url.lower() or 'openrouter' in self.source_config.url.lower():
            return {
                "model": self.model_config.model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        elif 'claude' in self.source_config.url.lower():
            return {
                "model": self.model_config.model_id,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens_to_sample": max_tokens
            }
        else:
            # 默认使用OpenAI格式
            return {
                "model": self.model_config.model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
    
    def _send_request(self, request_data: Dict[str, Any]) -> requests.Response:
        """发送HTTP请求"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.source_config.api_key}'
        }
        
        try:
            response = requests.post(
                self.source_config.url,
                headers=headers,
                json=request_data,
                timeout=300
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP请求失败: {str(e)}")
            raise AiServiceException(f"HTTP请求失败: {str(e)}")
    
    def _parse_response(self, response: requests.Response) -> str:
        """解析响应数据"""
        try:
            data = response.json()
            
            # 根据不同的AI服务提供商解析不同的响应格式
            if 'openai' in self.source_config.url.lower() or 'openrouter' in self.source_config.url.lower():
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    raise AiServiceException("响应格式错误：缺少choices字段")
            
            elif 'claude' in self.source_config.url.lower():
                if 'completion' in data:
                    return data['completion']
                else:
                    raise AiServiceException("响应格式错误：缺少completion字段")
            
            else:
                # 默认尝试解析OpenAI格式
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    raise AiServiceException("无法解析响应格式")
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}")
            raise AiServiceException(f"JSON解析失败: {str(e)}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用的模型列表"""
        models = AiModelConfig.objects.filter(is_active=True)
        return [
            {
                'id': model.id,
                'name': model.name,
                'model_type': model.get_model_type_display(),
                'source_name': model.source.name,
                'model_id': model.model_id,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature
            }
            for model in models
        ]
    
    def test_connection(self) -> bool:
        """测试AI服务连接"""
        try:
            test_prompt = "Hello, this is a test message."
            self.generate_text(test_prompt, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"AI服务连接测试失败: {str(e)}")
            return False


# 便捷函数
def generate_ai_text(prompt: str, model_name: str = None, **kwargs) -> str:
    """
    便捷函数：生成AI文本
    
    Args:
        prompt: 输入提示词
        model_name: 模型名称，可选
        **kwargs: 其他参数
        
    Returns:
        str: 生成的文本内容
    """
    service = AiService(model_name)
    return service.generate_text(prompt, **kwargs)


def get_ai_models() -> List[Dict[str, Any]]:
    """
    便捷函数：获取可用模型列表
    
    Returns:
        List[Dict]: 模型列表
    """
    service = AiService()
    return service.get_available_models()
