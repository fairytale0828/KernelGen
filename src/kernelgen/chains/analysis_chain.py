"""
分析链 - 使用LangChain实现的代码分析功能
"""

import json
import logging
from typing import Dict, Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.analysis_prompts import get_analysis_prompt

logger = logging.getLogger(__name__)

class AnalysisChain:
    """分析链 - 分析PyTorch代码并设计Triton kernel架构"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = get_analysis_prompt(is_initial=True)
        self.chain = self.prompt | llm | StrOutputParser()
    
    async def analyze_operation(self, pytorch_code: str, problem_info: Dict[str, Any], 
                              iteration: int = 1, previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析操作并生成架构设计
        
        Args:
            pytorch_code: PyTorch代码
            problem_info: 问题信息
            iteration: 迭代次数
            previous_results: 之前的结果（用于调试分析）
            
        Returns:
            分析结果
        """
        try:
            if iteration == 1:
                # 初始分析
                result = await self._initial_analysis(pytorch_code, problem_info)
            else:
                # 调试分析
                result = await self._debug_analysis(pytorch_code, previous_results)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _initial_analysis(self, pytorch_code: str, problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """初始分析"""
        
        # 准备输入数据
        input_data = {
            "pytorch_code": pytorch_code,
            "operation_name": problem_info.get("operation_name", "Unknown"),
            "input_shapes": str(problem_info.get("input_shapes", [])),
            "output_shapes": str(problem_info.get("output_shapes", []))
        }
        
        # 调用LLM
        response = await self.chain.ainvoke(input_data)
        
        # 解析JSON响应
        analysis_result = self._parse_analysis_response(response)
        analysis_result["analysis_type"] = "initial_analysis"
        
        return analysis_result
    
    async def _debug_analysis(self, pytorch_code: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """调试分析"""
        
        # 切换到调试分析提示
        debug_prompt = get_analysis_prompt(is_initial=False)
        debug_chain = debug_prompt | self.llm | StrOutputParser()
        
        # 准备输入数据
        input_data = {
            "pytorch_code": pytorch_code,
            "previous_code": previous_results.get("generated_code", ""),
            "error_info": previous_results.get("error_info", ""),
            "performance_info": json.dumps(previous_results.get("performance_info", {}), indent=2)
        }
        
        # 调用LLM
        response = await debug_chain.ainvoke(input_data)
        
        # 解析JSON响应
        analysis_result = self._parse_analysis_response(response)
        analysis_result["analysis_type"] = "debug_analysis"
        
        return analysis_result
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析分析响应"""
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
            
            # 如果都失败了，返回基本结构
            logger.warning("无法解析分析响应，使用默认结构")
            return {
                "operation_type": "unknown",
                "computational_pattern": "需要进一步分析",
                "memory_access_pattern": "需要进一步分析",
                "parallelization_strategy": "需要进一步分析",
                "block_size_recommendations": {
                    "BLOCK_SIZE": 256,
                    "reasoning": "默认推荐值"
                },
                "kernel_structure": {
                    "grid_dimensions": "需要根据输入大小确定",
                    "thread_organization": "一维线程块",
                    "memory_hierarchy": "全局内存访问"
                },
                "implementation_guidance": {
                    "key_optimizations": ["内存合并访问", "减少分支"],
                    "potential_challenges": ["边界处理", "数值精度"],
                    "triton_features": ["tl.load", "tl.store", "tl.program_id"]
                },
                "raw_response": response
            }