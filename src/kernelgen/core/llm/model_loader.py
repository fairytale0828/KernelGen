"""
模型加载器 - 负责创建和配置LLM模型
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
def get_project_root():
    """获取项目根目录的绝对路径"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# 缓存已加载的配置
_llm_config_cache = None


def load_llm_config() -> Dict[str, Any]:
    """加载LLM配置"""
    global _llm_config_cache
    
    if _llm_config_cache is not None:
        return _llm_config_cache
    
    try:
        config_path = os.path.join(get_project_root(), "core", "llm", "llm_config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            _llm_config_cache = yaml.safe_load(f)
        logger.info(f"加载LLM配置: {config_path}")
        return _llm_config_cache
    except Exception as e:
        logger.warning(f"加载LLM配置失败: {e}, 使用默认配置")
        _llm_config_cache = get_default_llm_config()
        return _llm_config_cache


def get_default_llm_config() -> Dict[str, Any]:
    """获取默认LLM配置"""
    return {
        "deepseek_default": {
            "model": "deepseek-chat",
            "api_base": "https://api.deepseek.com",
            "temperature": 0.1,
            "max_tokens": 4000,
            "timeout": 60
        },
        "deepseek_coder": {
            "model": "deepseek-coder",
            "api_base": "https://api.deepseek.com",
            "temperature": 0.05,
            "max_tokens": 8000,
            "timeout": 90
        }
    }


def create_model(model_name: str) -> ChatOpenAI:
    """
    创建LLM模型实例
    
    Args:
        model_name: 模型配置名称
        
    Returns:
        ChatOpenAI模型实例
    """
    llm_config = load_llm_config()
    
    # 获取模型配置
    if model_name in llm_config:
        model_config = llm_config[model_name]
    else:
        logger.warning(f"未找到模型配置: {model_name}, 使用默认配置")
        model_config = llm_config.get("deepseek_default", get_default_llm_config()["deepseek_default"])
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY环境变量未设置")
    
    # 创建模型实例
    model = ChatOpenAI(
        model=model_config.get("model", "deepseek-chat"),
        openai_api_key=api_key,
        openai_api_base=model_config.get("api_base", "https://api.deepseek.com"),
        temperature=model_config.get("temperature", 0.1),
        max_tokens=model_config.get("max_tokens", 4000),
        timeout=model_config.get("timeout", 60)
    )
    
    logger.debug(f"创建模型实例: {model_name}")
    return model