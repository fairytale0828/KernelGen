"""
Code Agent - 代码实现者
"""

import logging
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CodeAgent(BaseAgent):
    """
    代码Agent - 负责将设计方案转换为可执行的Triton代码
    
    职责：
    1. 根据设计方案生成Triton kernel代码
    2. 实现wrapper函数和辅助代码
    3. 确保代码语法正确和可编译
    4. 处理边界条件和错误处理
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("CodeAgent", llm_client, config)
    
    async def run(self, task_info: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        执行代码生成
        
        Args:
            task_info: 任务信息字典，包含设计方案和相关信息
            
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        try:
            # 从task_info中获取设计信息
            design_plan = task_info.get("design_plan", {})
            implementation_guide = task_info.get("implementation_guide", {})
            pytorch_code = task_info.get("pytorch_code", "")
            operator_analysis = task_info.get("operator_analysis", {})
            error_feedback = task_info.get("error_feedback", "")
            
            # 生成kernel代码
            kernel_code = await self._generate_kernel_code(
                pytorch_code, design_plan, implementation_guide, error_feedback
            )
            
            # 生成wrapper代码
            wrapper_code = await self._generate_wrapper_code(
                kernel_code, design_plan, operator_analysis
            )
            
            # 组合完整代码
            complete_code = self._combine_code_components(kernel_code, wrapper_code)
            
            # 验证代码
            validation_result = self._validate_generated_code(complete_code)
            
            if not validation_result["is_valid"]:
                logger.warning(f"生成的代码验证失败: {validation_result['error_message']}")
                # 尝试修复代码
                complete_code = self._fix_basic_issues(complete_code, validation_result)
            
            # 构造返回结果
            result_content = {
                "generated_code": complete_code,
                "kernel_code": kernel_code,
                "wrapper_code": wrapper_code,
                "validation_result": validation_result
            }
            
            # 格式化为JSON字符串
            import json
            formatted_result = json.dumps(result_content, indent=2, ensure_ascii=False)
            
            return formatted_result, "", ""
            
        except Exception as e:
            logger.error(f"代码生成失败: {str(e)}")
            raise
    
    async def _generate_kernel_code(self, pytorch_code: str, design_plan: Dict[str, Any],
                                  implementation_guide: Dict[str, Any], error_feedback: str) -> str:
        """生成kernel代码"""
        
        # 构造代码生成提示
        code_prompt = self._create_kernel_generation_prompt(
            pytorch_code, design_plan, implementation_guide, error_feedback
        )
        
        # 调用LLM生成代码
        llm_response = self.generate_llm_response(
            code_prompt,
            "You are an expert Triton kernel programmer. Generate efficient, correct Triton kernel code based on the design plan."
        )
        
        # 提取kernel代码
        kernel_code = self.extract_code_from_response(llm_response)
        
        return kernel_code
    
    async def _generate_wrapper_code(self, kernel_code: str, design_plan: Dict[str, Any],
                                   operator_analysis: Dict[str, Any]) -> str:
        """生成wrapper代码"""
        
        wrapper_prompt = self._create_wrapper_generation_prompt(
            kernel_code, design_plan, operator_analysis
        )
        
        llm_response = self.generate_llm_response(
            wrapper_prompt,
            "You are an expert in Triton kernel integration. Generate efficient wrapper functions."
        )
        
        wrapper_code = self.extract_code_from_response(llm_response)
        
        return wrapper_code
    
    def _combine_code_components(self, kernel_code: str, wrapper_code: str) -> str:
        """组合代码组件"""
        
        # 基础导入
        imports = """import torch
import triton
import triton.language as tl"""
        
        # 组合完整代码
        complete_code = f"""# Generated Triton Kernel

{imports}

{kernel_code}

{wrapper_code}
"""
        
        return complete_code
    
    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """验证生成的代码"""
        
        validation_result = {
            "is_valid": True,
            "error_message": "",
            "suggestions": [],
            "warnings": []
        }
        
        try:
            # 1. 语法检查
            compile(code, "<string>", "exec")
            
            # 2. 检查基本结构
            if "@triton.jit" not in code:
                validation_result["is_valid"] = False
                validation_result["error_message"] = "缺少@triton.jit装饰器"
            
            if "def " not in code:
                validation_result["is_valid"] = False
                validation_result["error_message"] = "缺少函数定义"
            
            # 3. 检查导入
            required_imports = ["import torch", "import triton"]
            for imp in required_imports:
                if imp not in code:
                    validation_result["warnings"].append(f"缺少导入: {imp}")
            
        except SyntaxError as e:
            validation_result["is_valid"] = False
            validation_result["error_message"] = f"语法错误: {str(e)}"
        
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["error_message"] = f"验证失败: {str(e)}"
        
        return validation_result
    
    def _fix_basic_issues(self, code: str, validation_result: Dict[str, Any]) -> str:
        """修复基础问题"""
        
        fixed_code = code
        
        # 修复缺少导入的问题
        if "import torch" not in fixed_code:
            fixed_code = "import torch\n" + fixed_code
        if "import triton" not in fixed_code:
            fixed_code = "import triton\n" + fixed_code
        if "import triton.language as tl" not in fixed_code:
            fixed_code = "import triton.language as tl\n" + fixed_code
        
        return fixed_code
    
    def _create_kernel_generation_prompt(self, pytorch_code: str, design_plan: Dict[str, Any],
                                       implementation_guide: Dict[str, Any], error_feedback: str) -> str:
        """创建kernel生成提示"""
        
        error_section = ""
        if error_feedback:
            error_section = f"\n## 错误反馈：\n{error_feedback}\n请根据错误反馈修复问题。"
        
        prompt = f"""
根据以下设计方案和实现指导，生成高效的Triton kernel代码：

## PyTorch参考代码：
```python
{pytorch_code}
```

## 设计方案：
{self._format_design_plan(design_plan)}

## 实现指导：
{self._format_implementation_guide(implementation_guide)}

{error_section}

## 代码要求：
1. 使用@triton.jit装饰器
2. 实现高效的内存访问
3. 正确处理边界条件
4. 包含必要的类型注解
5. 优化GPU并行性能

请生成完整的Triton kernel函数：
"""
        
        return prompt
    
    def _create_wrapper_generation_prompt(self, kernel_code: str, design_plan: Dict[str, Any],
                                        operator_analysis: Dict[str, Any]) -> str:
        """创建wrapper生成提示"""
        
        prompt = f"""
为以下Triton kernel生成wrapper函数：

## Kernel代码：
```python
{kernel_code}
```

## 设计信息：
- 算子类型: {operator_analysis.get('operator_type', 'unknown')}
- 推荐BLOCK_SIZE: {design_plan.get('recommended_block_size', 256)}

## Wrapper要求：
1. 处理输入张量的形状和设备
2. 计算合适的grid配置
3. 调用kernel函数
4. 返回正确的输出
5. 包含错误检查和边界处理

请生成完整的wrapper函数：
"""
        
        return prompt
    
    def _format_design_plan(self, design_plan: Dict[str, Any]) -> str:
        """格式化设计方案"""
        lines = []
        for key, value in design_plan.items():
            if key != "design_details":  # 排除详细描述
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _format_implementation_guide(self, implementation_guide: Dict[str, Any]) -> str:
        """格式化实现指导"""
        lines = []
        for key, value in implementation_guide.items():
            if key != "implementation_details":  # 排除详细描述
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)