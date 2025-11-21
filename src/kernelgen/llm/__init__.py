"""
LLM客户端模块 - 支持LangChain集成
"""

from .chat_models import DeepSeekChatModel, create_chat_model

__all__ = ["DeepSeekChatModel", "create_chat_model"]