# AI Manager 模块

AI Manager 是一个用于管理AI模型配置和提供标准化AI服务的Django应用模块。

## 功能特性

- **AI源配置管理**: 支持配置多个AI服务提供商（如OpenAI、Claude等）
- **AI模型配置管理**: 支持配置具体的AI模型参数
- **标准化AI服务接口**: 提供统一的AI文本生成接口
- **Web管理界面**: 提供完整的前端配置管理界面
- **连接测试**: 支持测试AI服务的连接状态

## 数据库表结构

### tb_ai_source_config (AI资源配置表)
- `name`: 资源名称
- `url`: 资源URL (如 https://openrouter.ai/api/v1/chat/completions)
- `api_key`: 密钥 (如 sk-XXX...)
- `is_active`: 是否启用
- `description`: 描述
- `created_at`: 创建时间
- `updated_at`: 更新时间

### tb_ai_model_config (AI模型配置表)
- `name`: 模型名称
- `model_type`: 模型类型 (1-文本模型，2-生图模型)
- `source`: 关联的AI源配置
- `model_id`: 模型ID
- `max_tokens`: 最大Token数
- `temperature`: 温度参数
- `is_active`: 是否启用
- `description`: 描述
- `created_at`: 创建时间
- `updated_at`: 更新时间

## API接口

### AI源配置管理
- `GET /webManager/ai/source/config` - 获取所有AI源配置
- `POST /webManager/ai/source/config` - 创建新的AI源配置
- `PUT /webManager/ai/source/config` - 更新AI源配置
- `DELETE /webManager/ai/source/config?id={id}` - 删除AI源配置

### AI模型配置管理
- `GET /webManager/ai/model/config` - 获取所有AI模型配置
- `POST /webManager/ai/model/config` - 创建新的AI模型配置
- `PUT /webManager/ai/model/config` - 更新AI模型配置
- `DELETE /webManager/ai/model/config?id={id}` - 删除AI模型配置

### AI服务接口
- `POST /webManager/ai/generate/text` - 生成AI文本
- `GET /webManager/ai/test/connection?model_name={name}` - 测试AI模型连接
- `GET /webManager/ai/available/models` - 获取可用模型列表

## 使用示例

### 1. 配置AI源

```python
from ai_manager.service.config_manager import AiConfigManager

# 创建AI源配置
source = AiConfigManager.create_source_config(
    name="OpenAI",
    url="https://api.openai.com/v1/chat/completions",
    api_key="sk-your-api-key",
    description="OpenAI GPT服务"
)
```

### 2. 配置AI模型

```python
# 创建AI模型配置
model = AiConfigManager.create_model_config(
    name="gpt-3.5-turbo",
    model_type=1,  # 文本模型
    source_id=source.id,
    model_id="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.7,
    description="GPT-3.5 Turbo模型"
)
```

### 3. 使用AI服务

```python
from ai_manager.service.ai_service import AiService, generate_ai_text

# 方式1: 使用服务类
service = AiService("gpt-3.5-turbo")
result = service.generate_text("你好，请介绍一下人工智能")

# 方式2: 使用便捷函数
result = generate_ai_text("你好，请介绍一下人工智能", model_name="gpt-3.5-turbo")
```

### 4. 前端使用

访问 `/ai-config` 页面可以：
- 管理AI源配置
- 管理AI模型配置
- 测试AI模型连接
- 在线测试AI文本生成

## 支持的AI服务提供商

目前支持以下AI服务提供商的API格式：
- OpenAI (包括OpenRouter)
- Claude (Anthropic)
- 其他兼容OpenAI格式的服务

## 注意事项

1. API密钥请妥善保管，不要泄露
2. 建议在生产环境中使用环境变量存储敏感信息
3. 定期测试AI服务的连接状态
4. 根据实际需求调整模型的温度参数和最大Token数
