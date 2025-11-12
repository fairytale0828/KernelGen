"""
生成链 - 使用LangChain实现的代码生成功能
"""

import json
import logging
from typing import Dict, Any

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.generation_prompts import get_generation_prompt, get_intelligent_fix_prompt
from ..prompts.analysis_prompts import get_hardware_context, get_knowledge_context

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
    
    async def fix_code_with_intelligent_analysis(self, pytorch_code: str, current_code: str, 
                                                error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基于智能错误分析进行硬件感知的代码修复"""
        try:
            intelligent_fix_prompt = get_intelligent_fix_prompt()
            intelligent_fix_chain = intelligent_fix_prompt | self.llm | StrOutputParser()
            
            # 获取硬件信息
            hardware_info = self._get_hardware_info()
            print("-------------------hardware_info--------------------")
            print(hardware_info)
            
            # 获取知识库信息
            operation_type = self._infer_operation_type(current_code)
            knowledge_context = self._get_knowledge_context(operation_type)
            
            input_data = {
                "pytorch_code": pytorch_code,
                "current_code": current_code,
                "error_analysis": f"{json.dumps(error_analysis, indent=2)}\n\n{hardware_info}\n\n{knowledge_context}"
            }
            
            response = await intelligent_fix_chain.ainvoke(input_data)
            fix_result = self._parse_fix_response(response)
            
            return {
                "success": True,
                "result": fix_result
            }
            
        except Exception as e:
            logger.error(f"基于智能分析的代码修复失败: {e}")
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
            print("-------------------device_info--------------------")
            print(device_info)
            
            return get_hardware_context(device_info)
        except Exception:
            return "## 硬件信息\n无法获取硬件信息"
    
    def _infer_operation_type(self, code: str) -> str:
        """从代码推断操作类型"""
        code_lower = code.lower()
        
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
                    "精确实现Conv2D + ReLU + BiasAdd序列",
                    "正确提取和使用PyTorch模型参数",
                    "处理4D张量的正确索引计算",
                    "确保数学等价性"
                ],
                "optimization_tips": [
                    "合理设计grid和block大小",
                    "优化4D张量内存访问",
                    "正确处理conv和额外bias",
                    "使用适当的数值精度"
                ]
            },
            "matmul": {
                "best_practices": ["使用分块算法", "优化内存访问", "利用共享内存"],
                "optimization_tips": ["BLOCK_SIZE平衡", "使用tl.dot", "内存合并访问"]
            },
            "convolution": {
                "best_practices": ["合理分块策略", "优化卷积核访问", "处理边界条件"],
                "optimization_tips": ["im2col算法", "共享内存缓存", "padding处理"]
            },
            "elementwise": {
                "best_practices": ["内存合并访问", "避免分支分歧", "向量化"],
                "optimization_tips": ["根据尺寸合理设置BLOCK_SIZE大小", "使用tl.where", "mask优化"]
            }
        }
        
        return get_knowledge_context(operation_type, {operation_type: knowledge_base.get(operation_type, {})})
    

    
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
            if match:
                kernel_code = match.group(1)
            else:
                # 如果没有找到代码块，尝试提取整个响应
                kernel_code = response.strip()
            
            # 确保代码包含必要的导入
            if kernel_code and not kernel_code.startswith('import'):
                kernel_code = "import torch\nimport triton\nimport triton.language as tl\n\n" + kernel_code
            
            # 如果代码为空或太短，记录警告
            if not kernel_code or len(kernel_code) < 50:
                logger.warning(f"生成的代码可能不完整，长度: {len(kernel_code)}")
                logger.warning(f"原始响应: {response[:500]}...")
            
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