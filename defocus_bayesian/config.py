"""
配置模块 - 加载和管理配置文件
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """
    配置管理类
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套路径
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        return self.config.get("model", {})
    
    def get_active_learning_config(self) -> Dict[str, Any]:
        """
        获取主动学习配置
        
        Returns:
            主动学习配置字典
        """
        return self.config.get("active_learning", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        获取可视化配置
        
        Returns:
            可视化配置字典
        """
        return self.config.get("visualization", {})
    
    def get_report_config(self) -> Dict[str, Any]:
        """
        获取报告配置
        
        Returns:
            报告配置字典
        """
        return self.config.get("report", {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        获取系统配置
        
        Returns:
            系统配置字典
        """
        return self.config.get("system", {})


# 创建全局配置实例
config = Config()
