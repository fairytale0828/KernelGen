"""
GeneratorAgent - 代码生成专家
基于aikg的Coder模式，负责根据AnalyzerAgent的设计生成Triton kernel代码
"""

import json
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class GeneratorAgent(BaseAgent):
    """
    代码生成专家Agent (基于aikg的Coder模式)
    
    职责：
    1. 根据AnalyzerAgent的架构设计生成Triton kernel代码
    2. 确保代码的正确性、可读性和性能优化
    3. 处理边界条件和数值稳定性问题
    4. 生成完整、可执行的代码实现
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("GeneratorAgent", llm_client, config)
        
        # 加载prompt模板
        self._load_prompt_templates()
        
        logger.info("GeneratorAgent初始化完成")
    
    def _load_prompt_templates(self):
        """加载prompt模板"""
        self.code_generation_template = """你是一个专业的Triton kernel编程专家，具有深厚的GPU编程知识和丰富的性能优化经验。

## 任务：根据架构设计生成Triton Kernel实现

### PyTorch参考代码（必须实现等价功能）：
```python
{pytorch_code}
```

### 架构设计方案：
{architecture_design}

### 实现指导：
{implementation_guidance}

### 任务信息：
- 问题ID: {problem_id}
- 操作名称: {operation_name}
- 输入形状: {input_shapes}
- 输出形状: {output_shapes}

### 代码生成要求：

1. **Kernel实现**：
   - 使用@triton.jit装饰器
   - 高效实现核心算法
   - 优化内存访问模式
   - 包含适当的边界检查

2. **包装函数**：
   - 创建用户友好的接口
   - 处理张量形状验证
   - 计算合适的grid配置
   - 管理设备放置
   - **重要**: 调用kernel时必须传递tensor的数据指针，使用tensor.data_ptr()而不是直接传递tensor对象

3. **代码质量**：
   - 添加全面的注释
   - 使用有意义的变量名
   - 遵循Triton最佳实践
   - 确保数值稳定性

4. **完整性要求**：
   - 包含kernel函数、包装函数和测试函数
   - 确保与PyTorch代码功能完全等价
   - 处理所有边界情况

5. **Triton调用规范**：
   - 直接传递tensor对象给kernel，不需要使用data_ptr()
   - 示例：kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
   - grid配置：grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
   - 确保输入tensor在CUDA设备上且连续：x.is_cuda 和 x.contiguous()

### 标准Triton代码模板：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def operation_kernel(
    x_ptr,  # 输入tensor指针
    out_ptr,  # 输出tensor指针
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前程序块的起始位置
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 执行操作
    out = operation(x)  # 替换为具体操作
    
    # 存储结果
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_operation(x: torch.Tensor):
    # 包装函数
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    operation_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

请按照以下JSON格式输出代码：

```json
{{
  "kernel_code": "完整的Triton kernel代码，包含kernel函数、包装函数和测试函数",
  "kernel_name": "kernel函数名称",
  "wrapper_name": "包装函数名称",
  "implementation_notes": "实现说明和关键优化点",
  "performance_hints": ["性能调优建议1", "性能调优建议2"]
}}
```"""

        self.code_fix_template = """你是一个专业的Triton kernel调试专家。

## 任务：根据错误信息修复Triton代码

### 原始PyTorch代码：
```python
{pytorch_code}
```

### 当前Triton代码：
```python
{current_code}
```

### 错误信息：
{error_info}

### 修复指导：
{fix_guidance}

### 修复要求：

1. **错误修复**：
   - 分析并修复编译错误
   - 解决运行时错误（特别是设备不匹配错误）
   - 修正逻辑错误

2. **常见错误修复**：
   - 设备错误：确保tensor在CUDA上且连续
   - 调用错误：直接传递tensor对象，不使用data_ptr()
   - Grid配置：使用lambda meta模式
   - 边界检查：正确使用mask和other参数

3. **代码改进**：
   - 保持代码结构清晰
   - 确保功能正确性
   - 提高代码健壮性

请按照以下JSON格式输出修复后的代码：

```json
{{
  "kernel_code": "修复后的完整Triton kernel代码",
  "kernel_name": "kernel函数名称",
  "wrapper_name": "包装函数名称",
  "fix_summary": "修复内容摘要",
  "changes_made": ["修改1", "修改2", "修改3"]
}}
```"""
    
    async def generate_code(self, 
                          pytorch_code: str,
                          problem_info: Dict[str, Any],
                          architecture_design: Dict[str, Any],
                          implementation_guidance: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据架构设计生成Triton代码
        
        Args:
            pytorch_code: PyTorch参考代码
            problem_info: 问题信息
            architecture_design: 架构设计
            implementation_guidance: 实现指导
            
        Returns:
            代码生成结果字典
        """
        try:
            input_data = {
                "pytorch_code": pytorch_code,
                "problem_id": problem_info.get("problem_id", ""),
                "operation_name": problem_info.get("operation_name", ""),
                "input_shapes": str(problem_info.get("input_shapes", [])),
                "output_shapes": str(problem_info.get("output_shapes", [])),
                "architecture_design": self._format_design_info(architecture_design),
                "implementation_guidance": self._format_guidance_info(implementation_guidance)
            }
            
            prompt = self.code_generation_template.format(**input_data)
            
            response = self.generate_llm_response(prompt)
            
            # 解析JSON响应
            result = self._extract_json_from_response(response)
            if result:
                logger.info("GeneratorAgent代码生成完成")
                return {
                    "success": True,
                    "result": result,
                    "raw_response": response
                }
            else:
                return {
                    "success": False,
                    "error": "响应解析失败",
                    "raw_response": response
                }
                
        except Exception as e:
            logger.error(f"GeneratorAgent执行失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def fix_code(self, 
                      pytorch_code: str,
                      current_code: str,
                      error_info: str,
                      fix_guidance: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据错误信息修复代码
        
        Args:
            pytorch_code: PyTorch参考代码
            current_code: 当前的Triton代码
            error_info: 错误信息
            fix_guidance: 修复指导
            
        Returns:
            代码修复结果字典
        """
        try:
            input_data = {
                "pytorch_code": pytorch_code,
                "current_code": current_code,
                "error_info": error_info,
                "fix_guidance": self._format_guidance_info(fix_guidance)
            }
            
            prompt = self.code_fix_template.format(**input_data)
            
            response = self.generate_llm_response(prompt)
            
            # 解析JSON响应
            result = self._extract_json_from_response(response)
            if result:
                logger.info("GeneratorAgent代码修复完成")
                return {
                    "success": True,
                    "result": result,
                    "raw_response": response
                }
            else:
                return {
                    "success": False,
                    "error": "响应解析失败",
                    "raw_response": response
                }
                
        except Exception as e:
            logger.error(f"GeneratorAgent代码修复失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_design_info(self, design: Dict[str, Any]) -> str:
        """格式化架构设计信息"""
        if not design:
            return "无架构设计信息"
        
        formatted = []
        for key, value in design.items():
            if isinstance(value, dict):
                formatted.append(f"**{key}**:")
                for sub_key, sub_value in value.items():
                    formatted.append(f"  - {sub_key}: {sub_value}")
            else:
                formatted.append(f"**{key}**: {value}")
        
        return "\n".join(formatted)
    
    def _format_guidance_info(self, guidance: Dict[str, Any]) -> str:
        """格式化实现指导信息"""
        if not guidance:
            return "无实现指导信息"
        
        formatted = []
        for key, value in guidance.items():
            if isinstance(value, list):
                formatted.append(f"**{key}**:")
                for item in value:
                    formatted.append(f"  - {item}")
            elif isinstance(value, dict):
                formatted.append(f"**{key}**:")
                for sub_key, sub_value in value.items():
                    formatted.append(f"  - {sub_key}: {sub_value}")
            else:
                formatted.append(f"**{key}**: {value}")
        
        return "\n".join(formatted)
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Agent的主要执行方法，兼容BaseAgent的抽象方法
        
        Args:
            **kwargs: 关键字参数
            
        Returns:
            代码生成结果字典
        """
        # 根据参数判断是生成代码还是修复代码
        if "architecture_design" in kwargs:
            # 生成代码模式
            pytorch_code = kwargs.get("pytorch_code", "")
            problem_info = kwargs.get("problem_info", {})
            architecture_design = kwargs.get("architecture_design", {})
            implementation_guidance = kwargs.get("implementation_guidance", {})
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self.generate_code(pytorch_code, problem_info, architecture_design, implementation_guidance)
            )
        else:
            # 修复代码模式
            pytorch_code = kwargs.get("pytorch_code", "")
            current_code = kwargs.get("current_code", "")
            error_info = kwargs.get("error_info", "")
            fix_guidance = kwargs.get("fix_guidance", {})
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self.fix_code(pytorch_code, current_code, error_info, fix_guidance)
            )
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """从响应中提取JSON"""
        try:
            # 查找JSON代码块
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                json_str = matches[0]
                return json.loads(json_str)
            
            # 尝试直接解析整个响应
            return json.loads(response)
            
        except Exception as e:
            logger.warning(f"JSON提取失败: {e}")
            return None