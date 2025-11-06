"""Base Agent类 - 提供强大的基础功能
基于aikg项目的设计，提供完整的Agent基础能力
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from pathlib import Path

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
        
        # 日志记录
        self.execution_logs = []
        self.performance_metrics = {}
        
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
        if self.llm_client:
            code = self.llm_client.extract_code_block(response_text, "python")
            if code:
                return code
        
        # 备用提取方法
        return self._extract_code_fallback(response_text)

    def _extract_code_fallback(self, text: str) -> str:
        """备用代码提取方法"""
        
        # 查找代码块标记
        markers = ["```python", "```triton", "```"]
        
        for marker in markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                # 跳过换行符
                while start < len(text) and text[start] in ['\n', '\r']:
                    start += 1
                end = text.find("```", start)
                if end != -1:
                    return text[start:end].strip()
        
        return text

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

    def create_simple_prompt(self, task_description: str, input_content: str, 
                           requirements: str = "", context: str = "") -> str:
        """
        创建简洁的prompt
        
        Args:
            task_description: 任务描述
            input_content: 输入内容
            requirements: 要求
            context: 上下文
            
        Returns:
            构建的prompt
        """
        prompt_parts = [task_description]
        
        if input_content:
            prompt_parts.append(f"\n## 输入:\n{input_content}")
        
        if context:
            prompt_parts.append(f"\n## 上下文:\n{context}")
            
        if requirements:
            prompt_parts.append(f"\n## 要求:\n{requirements}")
        
        return "\n".join(prompt_parts)

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析JSON响应"""
        try:
            return json.loads(response)
        except:
            # 如果不是JSON，返回包装的响应
            return {"content": response}

    def format_result_as_json(self, result_dict: Dict[str, Any]) -> str:
        """将结果格式化为JSON字符串"""
        return json.dumps(result_dict, indent=2, ensure_ascii=False)
    
    def log_execution(self, operation: str, input_data: Dict[str, Any], 
                     output_data: Dict[str, Any], success: bool, 
                     duration: float, error_msg: str = "") -> None:
        """记录执行日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "operation": operation,
            "success": success,
            "duration_ms": duration * 1000,
            "input_summary": self._summarize_data(input_data),
            "output_summary": self._summarize_data(output_data),
            "error_message": error_msg
        }
        self.execution_logs.append(log_entry)
        
        if success:
            logger.info(f"{self.agent_name} {operation} 成功: {duration:.3f}s")
        else:
            logger.error(f"{self.agent_name} {operation} 失败: {error_msg}")
    
    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """数据摘要，避免日志过大"""
        summary = {}
        for key, value in data.items():
            if isinstance(value, str):
                summary[key] = f"<string:{len(value)} chars>"
            elif isinstance(value, (list, tuple)):
                summary[key] = f"<{type(value).__name__}:{len(value)} items>"
            elif isinstance(value, dict):
                summary[key] = f"<dict:{len(value)} keys>"
            else:
                summary[key] = str(type(value).__name__)
        return summary
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        total_operations = len(self.execution_logs)
        successful_operations = len([log for log in self.execution_logs if log["success"]])
        total_duration = sum(log["duration_ms"] for log in self.execution_logs)
        
        return {
            "agent_name": self.agent_name,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / total_operations if total_operations > 0 else 0,
            "execution_logs": self.execution_logs
        }
    
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """从LLM响应中提取JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            import re
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 尝试查找JSON对象
            json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            return None
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: list) -> Tuple[bool, str]:
        """验证必需字段"""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"缺少必需字段: {missing_fields}"
        return True, ""
    
    def safe_llm_call(self, prompt: str, system_prompt: str = None, 
                     max_retries: int = 3) -> Tuple[bool, str, str]:
        """安全的LLM调用，包含重试和错误处理"""
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = self.generate_llm_response(prompt, system_prompt)
                duration = time.time() - start_time
                
                self.log_execution(
                    "llm_call",
                    {"prompt_length": len(prompt), "attempt": attempt + 1},
                    {"response_length": len(response)},
                    True,
                    duration
                )
                
                return True, response, ""
                
            except Exception as e:
                error_msg = str(e)
                if attempt == max_retries - 1:
                    duration = time.time() - start_time
                    self.log_execution(
                        "llm_call",
                        {"prompt_length": len(prompt), "attempts": max_retries},
                        {},
                        False,
                        duration,
                        error_msg
                    )
                    return False, "", error_msg
                
                logger.warning(f"{self.agent_name} LLM调用失败 (尝试 {attempt+1}/{max_retries}): {error_msg}")
                time.sleep(1)  # 等待后重试
        
        return False, "", "所有重试都失败了"

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Agent的主要执行方法，子类必须实现"""
        pass