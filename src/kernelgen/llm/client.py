"""
简化的LLM客户端，支持DeepSeek等API
"""

import os
import time
import logging
from typing import Dict, Any, Optional
import requests
import json

logger = logging.getLogger(__name__)

class LLMClient:
    """统一的LLM客户端"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化LLM客户端
        
        Args:
            llm_config: LLM配置
        """
        self.server_type = llm_config["server_type"]
        self.model_name = llm_config["model_name"]
        self.temperature = llm_config["temperature"]
        self.max_tokens = llm_config["max_tokens"]
        
        # 设置API密钥
        self.api_key = self._get_api_key()
        
        # 设置API端点
        self.api_endpoints = {
            "deepseek": "https://api.deepseek.com/v1/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions"
        }
        
        logger.info(f"初始化LLM客户端: {self.server_type}/{self.model_name}")
    
    def _get_api_key(self) -> str:
        """获取API密钥"""
        key_mapping = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        
        env_key = key_mapping.get(self.server_type)
        if not env_key:
            raise ValueError(f"不支持的服务器类型: {self.server_type}")
        
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"未设置环境变量: {env_key}")
        
        return api_key
    
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            
        Returns:
            生成的文本
        """
        if self.server_type in ["deepseek", "openai"]:
            return self._call_openai_compatible_api(prompt, system_prompt)
        else:
            raise ValueError(f"不支持的服务器类型: {self.server_type}")
    
    def _call_openai_compatible_api(self, prompt: str, system_prompt: str) -> str:
        """调用OpenAI兼容的API"""
        endpoint = self.api_endpoints[self.server_type]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.debug(f"LLM生成成功，长度: {len(content)}")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"API响应解析失败: {str(e)}")
            raise
    
    def extract_code_block(self, text: str, language: str = "python") -> Optional[str]:
        """
        从文本中提取代码块
        
        Args:
            text: 包含代码的文本
            language: 代码语言
            
        Returns:
            提取的代码，如果没找到返回None
        """
        # 查找markdown代码块
        import re
        
        # 匹配 ```python 或 ```cpp 等
        pattern = rf'```{language}\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 如果没找到特定语言的代码块，尝试匹配任意代码块
        pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 如果还是没找到，返回原文本（可能整个就是代码）
        return text.strip() if text.strip() else None