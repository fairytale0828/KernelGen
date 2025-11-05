"""
Base Agent类
"""

import os
import logging
from abc import ABC
from typing import Dict, Any, Tuple
import time

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Agent基类，提供基础功能和接口"""

    def __init__(self, agent_name: str, llm_client, config: Dict[str, Any]):
        """
        初始化Agent
        
        Args:
            agent_name: Agent名称
            llm_client: LLM客户端
            config: Agent配置
        """
        self.agent_name = agent_name
        self.llm_client = llm_client
        self.config = config
        self.step_count = 0
        
        logger.info(f"初始化{agent_name}")

    def generate_llm_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        调用LLM生成响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            
        Returns:
            LLM生成的文本
        """
        if system_prompt is None:
            system_prompt = f"You are a {self.agent_name} specialized in GPU kernel development."
        
        try:
            response = self.llm_client.generate(prompt, system_prompt)
            return response
        except Exception as e:
            logger.error(f"{self.agent_name} LLM调用失败: {str(e)}")
            raise

    def extract_code_from_response(self, response_text: str) -> str:
        """
        从LLM响应中提取代码
        
        Args:
            response_text: LLM响应文本
            
        Returns:
            提取的代码
        """
        code = self.llm_client.extract_code_block(response_text, "python")
        return code if code else response_text

    async def run_llm(self, prompt_text: str, input_data: Dict[str, Any], 
                     model_name: str) -> Tuple[str, str, str]:
        """
        运行LLM生成
        
        Args:
            prompt_text: 格式化的提示文本
            input_data: 输入数据
            model_name: 模型名称
            
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        try:
            # 格式化提示
            formatted_prompt = prompt_text.format(**input_data)
            
            # 调用LLM
            content = self.generate_llm_response(formatted_prompt)
            
            # 返回兼容格式
            return content, formatted_prompt, ""
            
        except Exception as e:
            logger.error(f"{self.agent_name} LLM执行失败: {str(e)}")
            raise