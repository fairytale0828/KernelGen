"""
生成链 - 使用LangChain实现的代码生成功能
"""

import json
import logging
from typing import Dict, Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.generation_prompts import get_generation_prompt, get_intelligent_fix_prompt
from ..services import HardwareInfoService, KnowledgeBaseService, OperationTypeService, CodeValidationService

logger = logging.getLogger(__name__)

class GenerationChain:
    """生成链 - 生成Triton kernel代码"""
    
    def __init__(self, llm: BaseChatModel,
                 hardware_service: Optional[HardwareInfoService] = None,
                 knowledge_service: Optional[KnowledgeBaseService] = None,
                 operation_service: Optional[OperationTypeService] = None,
                 validation_service: Optional[CodeValidationService] = None):
        self.llm = llm
        
        # 初始化服务
        self.hardware_service = hardware_service or HardwareInfoService()
        self.knowledge_service = knowledge_service or KnowledgeBaseService()
        self.operation_service = operation_service or OperationTypeService(llm)
        self.validation_service = validation_service or CodeValidationService()
        
        # 初始化prompt和chain
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
                                                error_analysis: Dict[str, Any], 
                                                cached_operation_type: str = None) -> Dict[str, Any]:
        """统一的智能代码修复接口"""
        try:
            # 1. 使用缓存的操作类型或推断操作类型
            if cached_operation_type:
                operation_type = cached_operation_type
            else:
                operation_result = await self.operation_service.infer_operation_type(current_code)
                operation_type = "unknown"
                if operation_result.get("success"):
                    operation_type = operation_result["result"].get("primary_operation", "unknown")
                logger.info(f"推断操作类型: {operation_type}")
            
            # 2. 获取上下文信息
            hardware_context = self.hardware_service.get_hardware_context_string()
            knowledge_context = self.knowledge_service.get_knowledge_context_string(operation_type)
            
            # 3. 使用智能修复提示
            intelligent_fix_prompt = get_intelligent_fix_prompt()
            intelligent_fix_chain = intelligent_fix_prompt | self.llm | StrOutputParser()
            
            input_data = {
                "pytorch_code": pytorch_code,
                "current_code": current_code,
                "error_analysis": f"{json.dumps(error_analysis, indent=2)}\n\n{hardware_context}\n\n{knowledge_context}"
            }
            
            # 4. 调用LLM生成修复代码
            response = await intelligent_fix_chain.ainvoke(input_data)
            fix_result = self._parse_fix_response(response)
            
            # 5. 验证生成的代码
            if fix_result.get("fixed_kernel_code") or fix_result.get("kernel_code"):
                code_to_validate = fix_result.get("fixed_kernel_code") or fix_result.get("kernel_code")
                validation_result = self.validation_service.validate_triton_code(code_to_validate)
                fix_result["validation_result"] = {
                    "is_valid": validation_result.is_valid,
                    "issues": validation_result.issues,
                    "warnings": validation_result.warnings,
                    "quality_score": validation_result.score
                }
                
                if not validation_result.is_valid:
                    logger.warning(f"生成的修复代码存在验证问题: {validation_result.issues}")
            
            return {
                "success": True,
                "result": fix_result
            }
            
        except Exception as e:
            logger.error(f"智能代码修复失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # 注意：硬件信息、操作类型推断、知识库上下文现在由services模块提供
    # 这些重复的方法已被移除，统一使用services中的实现
    
    def _parse_generation_response(self, response: str) -> Dict[str, Any]:
        """解析生成响应 - 增强版"""
        try:
            # 尝试直接解析JSON
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            import re
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Fallback: 尝试提取Python代码块
            kernel_code = self._extract_code_from_response(response)
            
            result = {
                "kernel_code": kernel_code,
                "kernel_name": "generated_kernel",
                "wrapper_name": self._extract_wrapper_name(kernel_code),
                "launch_config": {
                    "grid_function": "lambda meta: (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']),)",
                    "block_sizes": {"BLOCK_SIZE": 256}
                },
                "usage_example": "# 使用示例需要根据具体kernel调整",
                "optimization_notes": ["从响应中提取的代码"],
                "raw_response": response,
                "parsing_method": "fallback"
            }
            
            # 代码验证由validation_chain统一处理
            
            logger.warning("使用fallback方法解析生成响应")
            return result
    
    def _extract_code_from_response(self, response: str) -> str:
        """从响应中提取代码"""
        import re
        
        # 尝试提取Python代码块
        python_pattern = r'```python\s*(.*?)\s*```'
        match = re.search(python_pattern, response, re.DOTALL)
        if match:
            kernel_code = match.group(1)
        else:
            # 如果没有找到代码块，尝试提取整个响应
            kernel_code = response.strip()
        
        # 确保代码包含必要的导入
        if kernel_code and not any(imp in kernel_code for imp in ["import torch", "import triton"]):
            kernel_code = "import torch\nimport triton\nimport triton.language as tl\n\n" + kernel_code
        
        return kernel_code
    
    def _extract_wrapper_name(self, code: str) -> str:
        """从代码中提取wrapper函数名"""
        import re
        
        # 查找接受torch.Tensor参数的函数
        pattern = r'def\s+(\w+)\s*\([^)]*torch\.Tensor[^)]*\)'
        match = re.search(pattern, code)
        if match:
            return match.group(1)
        
        # 查找不是@triton.jit装饰的函数
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and i > 0:
                prev_line = lines[i-1].strip()
                if not prev_line.startswith('@triton.jit'):
                    match = re.search(r'def\s+(\w+)', line)
                    if match:
                        return match.group(1)
        
        return "triton_wrapper"
    
    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """验证生成的代码"""
        try:
            validation_result = self.validation_service.validate_triton_code(code)
            return {
                "is_valid": validation_result.is_valid,
                "issues": validation_result.issues,
                "warnings": validation_result.warnings,
                "suggestions": validation_result.suggestions,
                "quality_score": validation_result.score
            }
        except Exception as e:
            logger.error(f"代码验证失败: {e}")
            return {
                "is_valid": False,
                "issues": [f"验证过程出错: {str(e)}"],
                "warnings": [],
                "suggestions": [],
                "quality_score": 0.0
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