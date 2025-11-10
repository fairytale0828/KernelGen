"""
生成链 - 使用LangChain实现的代码生成功能
"""

import json
import logging
from typing import Dict, Any

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.generation_prompts import get_generation_prompt

logger = logging.getLogger(__name__)

class GenerationChain:
    """生成链 - 生成Triton kernel代码"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = get_generation_prompt(is_initial=True)
        self.chain = self.prompt | llm | StrOutputParser()
    
    async def generate_code(self, pytorch_code: str, problem_info: Dict[str, Any],
                          architecture_design: Dict[str, Any], 
                          implementation_guidance: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成Triton kernel代码
        
        Args:
            pytorch_code: PyTorch参考代码
            problem_info: 问题信息
            architecture_design: 架构设计
            implementation_guidance: 实现指导
            
        Returns:
            生成结果
        """
        try:
            # 准备输入数据
            input_data = {
                "pytorch_code": pytorch_code,
                "architecture_design": json.dumps(architecture_design, indent=2, ensure_ascii=False),
                "implementation_guidance": json.dumps(implementation_guidance, indent=2, ensure_ascii=False)
            }
            
            # 调用LLM
            response = await self.chain.ainvoke(input_data)
            
            # 解析响应
            generation_result = self._parse_generation_response(response)
            
            return {
                "success": True,
                "result": generation_result
            }
            
        except Exception as e:
            logger.error(f"代码生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def fix_code(self, pytorch_code: str, current_code: str, 
                      error_info: str, fix_guidance: Dict[str, Any]) -> Dict[str, Any]:
        """
        修复Triton kernel代码
        
        Args:
            pytorch_code: PyTorch参考代码
            current_code: 当前有问题的代码
            error_info: 错误信息
            fix_guidance: 修复指导
            
        Returns:
            修复结果
        """
        try:
            # 切换到修复生成提示
            fix_prompt = get_generation_prompt(is_initial=False)
            fix_chain = fix_prompt | self.llm | StrOutputParser()
            
            # 准备输入数据
            input_data = {
                "pytorch_code": pytorch_code,
                "current_code": current_code,
                "error_info": error_info,
                "fix_guidance": json.dumps(fix_guidance, indent=2, ensure_ascii=False)
            }
            
            # 调用LLM
            response = await fix_chain.ainvoke(input_data)
            
            # 解析响应
            fix_result = self._parse_fix_response(response)
            
            return {
                "success": True,
                "result": fix_result
            }
            
        except Exception as e:
            logger.error(f"代码修复失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_generation_response(self, response: str) -> Dict[str, Any]:
        """解析生成响应"""
        try:
            # 尝试直接解析JSON
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
            
            # 尝试提取Python代码块作为kernel_code
            python_pattern = r'```python\s*(.*?)\s*```'
            match = re.search(python_pattern, response, re.DOTALL)
            kernel_code = match.group(1) if match else response
            
            # 确保代码包含必要的导入
            if kernel_code and not kernel_code.startswith('import'):
                kernel_code = "import torch\nimport triton\nimport triton.language as tl\n\n" + kernel_code
            
            logger.warning("无法解析生成响应，提取代码块")
            return {
                "kernel_code": kernel_code,
                "kernel_name": "generated_kernel",
                "launch_config": {
                    "grid_function": "lambda meta: (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']),)",
                    "block_sizes": {"BLOCK_SIZE": 256}
                },
                "usage_example": "# 使用示例需要根据具体kernel调整",
                "optimization_notes": ["从响应中提取的代码"],
                "raw_response": response
            }
    
    def _parse_fix_response(self, response: str) -> Dict[str, Any]:
        """解析修复响应"""
        try:
            # 尝试直接解析JSON
            parsed = json.loads(response)
            # 将fixed_kernel_code映射到kernel_code以保持一致性
            if "fixed_kernel_code" in parsed:
                parsed["kernel_code"] = parsed["fixed_kernel_code"]
            return parsed
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            import re
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if "fixed_kernel_code" in parsed:
                        parsed["kernel_code"] = parsed["fixed_kernel_code"]
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取Python代码块
            python_pattern = r'```python\s*(.*?)\s*```'
            match = re.search(python_pattern, response, re.DOTALL)
            kernel_code = match.group(1) if match else response
            
            # 确保代码包含必要的导入
            if kernel_code and not kernel_code.startswith('import'):
                kernel_code = "import torch\nimport triton\nimport triton.language as tl\n\n" + kernel_code
            
            logger.warning("无法解析修复响应，提取代码块")
            return {
                "kernel_code": kernel_code,
                "kernel_name": "fixed_kernel",
                "changes_made": ["从响应中提取的修复代码"],
                "fix_reasoning": {
                    "error_fixes": "尝试修复识别的错误",
                    "performance_improvements": "应用常见优化",
                    "stability_enhancements": "增强稳定性"
                },
                "launch_config": {
                    "grid_function": "lambda meta: (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']),)",
                    "block_sizes": {"BLOCK_SIZE": 256}
                },
                "raw_response": response
            }