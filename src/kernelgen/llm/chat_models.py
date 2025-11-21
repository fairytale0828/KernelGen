"""
LangChain兼容的聊天模型封装
"""

import os
import logging
from typing import Any, Dict, List, Optional
import requests
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)

class DeepSeekChatModel(BaseChatModel):
    """DeepSeek API的LangChain封装"""
    
    model_name: str = "deepseek-coder"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY环境变量未设置")
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成聊天响应"""
        
        # 转换消息格式
        api_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                api_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                api_messages.append({"role": "assistant", "content": message.content})
        
        # 调用API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120  # 增加超时时间到120秒
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 创建ChatGeneration
            generation = ChatGeneration(message=AIMessage(content=content))
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise

def create_chat_model(config: Dict[str, Any]) -> BaseChatModel:
    """根据配置创建聊天模型"""
    server_type = config.get("server_type", "deepseek")
    
    if server_type == "deepseek":
        return DeepSeekChatModel(
            model_name=config.get("model_name", "deepseek-coder"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 4096)
        )
    else:
        raise ValueError(f"不支持的服务器类型: {server_type}")