"""
分析链 - 使用LangChain实现的代码分析功能
"""

import json
import logging
from typing import Dict, Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.analysis_prompts import get_analysis_prompt, get_error_analysis_prompt
from ..services import HardwareInfoService, KnowledgeBaseService, OperationTypeService

logger = logging.getLogger(__name__)

class AnalysisChain:
    """分析链 - 分析PyTorch代码并设计Triton kernel架构"""
    
    def __init__(self, llm: BaseChatModel, 
                 hardware_service: Optional[HardwareInfoService] = None,
                 knowledge_service: Optional[KnowledgeBaseService] = None,
                 operation_service: Optional[OperationTypeService] = None):
        self.llm = llm
        
        # 初始化服务
        self.hardware_service = hardware_service or HardwareInfoService()
        self.knowledge_service = knowledge_service or KnowledgeBaseService()
        self.operation_service = operation_service or OperationTypeService(llm)
        
        # 初始化prompt和chain
        self.prompt = get_analysis_prompt(is_initial=True)
        self.chain = self.prompt | llm | StrOutputParser()
    
    async def analyze_operation(self, pytorch_code: str, problem_info: Dict[str, Any], 
                              iteration: int = 1, previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        统一的操作分析接口
        
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
                # 错误分析和修复建议
                result = await self._error_analysis(pytorch_code, previous_results)
            
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
        
        # 1. 推断操作类型
        operation_result = await self.operation_service.infer_operation_type(
            pytorch_code, 
            problem_info.get("operation_name", "")
        )
        
        operation_type = "unknown"
        if operation_result.get("success"):
            operation_type = operation_result["result"].get("primary_operation", "unknown")
        elif operation_result.get("fallback_result"):
            operation_type = operation_result["fallback_result"].get("primary_operation", "unknown")
        
        logger.info(f"推断的操作类型: {operation_type}")
        
        # 2. 获取硬件信息和知识库
        hardware_context = self.hardware_service.get_hardware_context_string()
        knowledge_context = self.knowledge_service.get_knowledge_context_string(operation_type)
        
        # 3. 准备输入数据
        input_data = {
            "pytorch_code": pytorch_code,
            "operation_name": problem_info.get("operation_name", "Unknown"),
            "input_shapes": str(problem_info.get("input_shapes", [])),
            "output_shapes": str(problem_info.get("output_shapes", [])),
            "hardware_context": hardware_context,
            "knowledge_context": knowledge_context,
            "operation_type": operation_type
        }
        
        # 4. 调用LLM
        response = await self.chain.ainvoke(input_data)
        
        # 5. 解析JSON响应
        analysis_result = self._parse_analysis_response(response, {})
        analysis_result["analysis_type"] = "initial_analysis"
        analysis_result["inferred_operation_type"] = operation_type
        
        return analysis_result
    
    async def _error_analysis(self, pytorch_code: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """统一的错误分析"""
        
        # 1. 推断操作类型（从之前的代码）
        previous_code = previous_results.get("generated_code", "")
        operation_result = await self.operation_service.infer_operation_type(previous_code)
        
        operation_type = "unknown"
        if operation_result.get("success"):
            operation_type = operation_result["result"].get("primary_operation", "unknown")
        
        # 2. 获取上下文信息
        hardware_context = self.hardware_service.get_hardware_context_string()
        knowledge_context = self.knowledge_service.get_knowledge_context_string(operation_type)
        
        # 3. 使用错误分析提示
        error_analysis_prompt = get_error_analysis_prompt()
        error_analysis_chain = error_analysis_prompt | self.llm | StrOutputParser()
        
        # 4. 准备输入数据
        error_info = previous_results.get("error_info", "")
        performance_data = previous_results.get("performance_info", {})
        
        input_data = {
            "error_info": f"{error_info}\n\n{hardware_context}\n\n{knowledge_context}",
            "code_context": previous_code,
            "performance_data": json.dumps(performance_data, indent=2)
        }
        
        # 5. 调用LLM
        response = await error_analysis_chain.ainvoke(input_data)
        
        # 6. 解析响应
        analysis_result = self._parse_analysis_response(response, previous_results)
        analysis_result["analysis_type"] = "error_analysis"
        analysis_result["operation_type"] = operation_type
        
        return analysis_result
    
    def get_operation_recommendations(self, operation_type: str) -> Dict[str, Any]:
        """获取操作类型相关的推荐配置"""
        hardware_recommendations = self.hardware_service.get_optimization_recommendations(operation_type)
        knowledge = self.knowledge_service.get_knowledge(operation_type)
        
        recommendations = {
            "hardware_optimizations": hardware_recommendations,
            "implementation_patterns": knowledge.implementation_patterns if knowledge else [],
            "performance_considerations": knowledge.performance_considerations if knowledge else []
        }
        
        return recommendations
    
    async def analyze_error_intelligently(self, error_info: str, code_context: str, 
                                        performance_data: Dict[str, Any], pytorch_code: str = "") -> Dict[str, Any]:
        """智能错误分析 - 统一接口"""
        try:
            # 构建previous_results格式以复用现有逻辑
            previous_results = {
                "generated_code": code_context,
                "error_info": error_info,
                "performance_info": performance_data
            }
            
            # 调用现有的错误分析方法
            result = await self._error_analysis(pytorch_code, previous_results)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"智能错误分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": {
                    "analysis_type": "error_analysis",
                    "error_categories": ["分析失败"],
                    "root_causes": [f"分析过程出错: {str(e)}"],
                    "fix_recommendations": ["请检查代码和错误信息"],
                    "implementation_guidance": {
                        "key_optimizations": ["基本优化"],
                        "potential_challenges": ["错误分析失败"],
                        "triton_features": ["基本Triton功能"]
                    }
                }
            }
    
    def _parse_analysis_response(self, response: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
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