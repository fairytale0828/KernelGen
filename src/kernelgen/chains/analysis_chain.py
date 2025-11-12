"""
分析链 - 使用LangChain实现的代码分析功能
"""

import json
import logging
from typing import Dict, Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.analysis_prompts import get_analysis_prompt, get_error_analysis_prompt, get_hardware_context, get_knowledge_context

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
        # print("-------------------input_data--------------------")
        # print(input_data)
        
        # 调用LLM
        response = await self.chain.ainvoke(input_data)
        # print("-------------------response--------------------")
        # print(response)
        
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
    
    async def analyze_error_intelligently(self, error_info: str, code_context: str, 
                                        performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """智能错误分析 - 硬件感知的深度分析"""
        try:
            error_analysis_prompt = get_error_analysis_prompt()
            error_analysis_chain = error_analysis_prompt | self.llm | StrOutputParser()
            
            # 获取硬件信息
            hardware_info = self._get_hardware_info()
            
            # 获取知识库信息
            operation_type = self._infer_operation_type(code_context)
            knowledge_context = self._get_knowledge_context(operation_type)
            
            input_data = {
                "error_info": f"{error_info}\n\n{hardware_info}\n\n{knowledge_context}",
                "code_context": code_context,
                "performance_data": json.dumps(performance_data or {}, indent=2)
            }
            
            response = await error_analysis_chain.ainvoke(input_data)
            analysis_result = self._parse_analysis_response(response)
            analysis_result["analysis_type"] = "intelligent_error_analysis"
            
            return {
                "success": True,
                "result": analysis_result
            }
            
        except Exception as e:
            logger.error(f"智能错误分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_hardware_info(self) -> str:
        """获取硬件信息"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                device_info = {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_size": f"{props.total_memory / 1024**3:.1f}GB",
                    "sm_count": props.multi_processor_count
                }
            else:
                device_info = {"name": "CPU", "compute_capability": "N/A"}
            
            return get_hardware_context(device_info)
        except Exception:
            return "## 硬件信息\n无法获取硬件信息"
    
    def _infer_operation_type(self, code_context: str) -> str:
        """从代码上下文推断操作类型"""
        code_lower = code_context.lower()
        
        if any(pattern in code_lower for pattern in ["conv2d", "conv_relu", "conv.*relu.*bias"]):
            return "conv2d_composite"
        elif any(pattern in code_lower for pattern in ["matmul", "mm", "bmm", "dot"]):
            return "matmul"
        elif any(pattern in code_lower for pattern in ["conv", "convolution"]):
            return "convolution"
        elif any(pattern in code_lower for pattern in ["relu", "sigmoid", "tanh", "add", "mul"]):
            return "elementwise"
        elif any(pattern in code_lower for pattern in ["sum", "mean", "max", "min", "softmax"]):
            return "reduction"
        else:
            return "unknown"
    
    def _get_knowledge_context(self, operation_type: str) -> str:
        """获取知识库上下文"""
        knowledge_base = {
            "conv2d_composite": {
                "best_practices": [
                    "分析PyTorch模型的每个组件：Conv2D + ReLU + BiasAdd",
                    "正确处理4D张量索引：(batch, channel, height, width)",
                    "区分conv内置bias和额外bias参数",
                    "确保与PyTorch nn.Conv2d的数学等价性"
                ],
                "common_issues": [
                    "忽略batch维度导致索引错误",
                    "混淆conv内置bias和额外bias",
                    "4D张量展平和重构错误",
                    "padding和stride计算错误"
                ],
                "optimization_tips": [
                    "使用合适的BLOCK_SIZE处理输出元素",
                    "优化内存访问模式避免bank conflicts",
                    "正确处理边界条件和padding"
                ]
            },
            "matmul": {
                "best_practices": ["使用分块算法", "优化内存访问", "利用共享内存"],
                "common_issues": ["分块大小不当", "内存访问不连续"],
                "optimization_tips": ["BLOCK_SIZE平衡", "使用tl.dot"]
            },
            "convolution": {
                "best_practices": ["合理分块策略", "优化卷积核访问", "处理边界条件"],
                "common_issues": ["边界处理复杂", "内存访问不优化"],
                "optimization_tips": ["im2col算法", "共享内存缓存"]
            },
            "elementwise": {
                "best_practices": ["内存合并访问", "避免分支分歧", "向量化"],
                "common_issues": ["mask使用不当", "BLOCK_SIZE不合理"],
                "optimization_tips": ["BLOCK_SIZE=256/512", "使用tl.where"]
            }
        }
        
        return get_knowledge_context(operation_type, {operation_type: knowledge_base.get(operation_type, {})})
    

    
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