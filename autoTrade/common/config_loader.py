# common/config_loader.py

import json
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        config_path = os.path.join(settings.BASE_DIR, 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            logger.info("ConfigLoader: config.json 加载成功。")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(f"ConfigLoader: 无法加载或解析 config.json: {e}。系统将无法正常运行。")
            self._config = {} # 返回一个空字典以避免后续调用出错

    def get_config(self):
        """获取完整的配置字典"""
        return self._config

    def get(self, key, default=None):
        """获取指定键的配置值"""
        return self._config.get(key, default)

# 创建一个全局实例，供项目各处调用
config_loader = ConfigLoader()
