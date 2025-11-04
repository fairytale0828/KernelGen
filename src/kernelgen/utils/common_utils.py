"""
通用工具函数
"""

import os
def get_project_root():
    """获取项目根目录的绝对路径"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_prompt_path() -> str:
    """
    获取提示词模板目录路径
    
    Returns:
        提示词模板目录的绝对路径
    """
    return os.path.join(get_project_root(), "resources", "prompts")


def get_docs_path() -> str:
    """
    获取文档资源目录路径
    
    Returns:
        文档资源目录的绝对路径
    """
    return os.path.join(get_project_root(), "resources", "docs")


def get_config_path() -> str:
    """
    获取配置文件目录路径
    
    Returns:
        配置文件目录的绝对路径
    """
    return os.path.join(get_project_root(), "config")