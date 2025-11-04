"""
配置管理器 - 负责加载和管理系统配置
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器
    
    负责加载和管理系统配置，包括API配置、模型参数、
    验证设置等各种配置项。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化配置管理器
        
        Args:
            config: 外部传入的配置，如果为None则使用默认配置
        """
        self.config = self._load_default_config()
        
        if config:
            self.config.update(config)
        
        # 从环境变量加载API配置
        self._load_env_config()
        
        logger.info("配置管理器初始化完成")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "deepseek_api": {
                "model": "deepseek-chat",
                "api_key": None,
                "api_base": "https://api.deepseek.com",
                "temperature": 0.1,
                "max_tokens": 4000,
                "timeout": 60
            },
            "verification": {
                "num_warmup": 10,
                "num_runs": 100,
                "timeout": 300
            },
            "generation": {
                "max_iterations": 5,
                "enable_performance_test": True,
                "enable_functionality_test": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _load_env_config(self):
        """从环境变量加载配置"""
        # DeepSeek API配置
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            self.config["deepseek_api"]["api_key"] = api_key
            logger.info("从环境变量加载DeepSeek API密钥")
        
        api_base = os.getenv("DEEPSEEK_API_BASE")
        if api_base:
            self.config["deepseek_api"]["api_base"] = api_base
            logger.info(f"从环境变量加载DeepSeek API地址: {api_base}")
        
        # 日志级别
        log_level = os.getenv("KERNELGEN_LOG_LEVEL")
        if log_level:
            self.config["logging"]["level"] = log_level.upper()
            logger.info(f"从环境变量设置日志级别: {log_level}")
        
        # 验证超时时间
        timeout = os.getenv("KERNELGEN_VERIFY_TIMEOUT")
        if timeout:
            try:
                self.config["verification"]["timeout"] = int(timeout)
                logger.info(f"从环境变量设置验证超时: {timeout}秒")
            except ValueError:
                logger.warning(f"无效的超时时间设置: {timeout}")
    
    def load_from_file(self, config_path: str):
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    import json
                    file_config = json.load(f)
            
            self.config.update(file_config)
            logger.info(f"从文件加载配置: {config_path}")
            
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
    
    def get_deepseek_config(self) -> Dict[str, Any]:
        """获取DeepSeek API配置"""
        return self.config["deepseek_api"].copy()
    
    def get_verification_config(self) -> Dict[str, Any]:
        """获取验证配置"""
        return self.config["verification"].copy()
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return self.config["generation"].copy()
    
    def validate_config(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            配置是否有效
        """
        # 检查必要的API密钥
        if not self.config["deepseek_api"]["api_key"]:
            logger.error("DeepSeek API密钥未配置")
            return False
        
        # 检查数值配置的合理性
        if self.config["verification"]["timeout"] <= 0:
            logger.error("验证超时时间必须大于0")
            return False
        
        if self.config["generation"]["max_iterations"] <= 0:
            logger.error("最大迭代次数必须大于0")
            return False
        
        logger.info("配置验证通过")
        return True
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 要更新的配置项
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        logger.info("配置已更新")
    
    def save_to_file(self, config_path: str):
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                else:
                    import json
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")