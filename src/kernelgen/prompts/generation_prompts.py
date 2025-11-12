"""
代码生成阶段的LangChain提示模板
"""

from langchain_core.prompts import PromptTemplate

# 初始生成提示模板
INITIAL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["pytorch_code", "architecture_design", "implementation_guidance"],
    template="""你是一个专业的Triton kernel开发专家。请根据架构设计生成高性能的Triton kernel。

## PyTorch参考代码（来自KernelBench数据集）
```python
{pytorch_code}
```

## 重要说明
这个PyTorch代码包含：
1. `get_inputs()`: 返回测试输入张量列表
2. `get_init_inputs()`: 返回模型初始化参数
3. `Model`类: 实现了要转换为Triton的算子逻辑
4. 你需要实现与Model.forward()等价的Triton kernel

## Triton函数接口设计原则
- 必须仔细分析PyTorch模型的参数结构，确定Triton函数需要的所有参数
- 例如：如果Model有conv层和额外bias，Triton函数应该是 `triton_func(x, conv_weight, conv_bias, extra_bias)`
- **关键**：Triton实现必须与PyTorch Model.forward()在数学上完全等价
- 性能测试工具会自动从PyTorch模型中提取参数，传递给Triton函数

## 参数提取和使用规则
1. **分析模型结构**：识别所有nn.Module层及其参数
2. **理解参数含义**：区分不同类型的权重和偏置
3. **设计函数签名**：确保Triton函数接收所有必要参数
4. **保证数值等价**：使用相同参数确保计算结果一致

## 架构设计
{architecture_design}

## 实现指导
{implementation_guidance}

## 完整Triton实现模式（必须包含kernel和wrapper函数）
```python
import torch
import triton
import triton.language as tl

@triton.jit
def example_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # 在这里进行计算
    result = x  # 替换为实际计算
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_example(x: torch.Tensor):
    \"\"\"Wrapper函数 - 必须包含\"\"\"
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    example_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

# 注意：不要生成测试函数！
# 测试由KernelGen系统自动完成，使用KernelBench原生PyTorch模型
```

## 生成要求
1. **精确分析PyTorch模型**：仔细分析架构设计中的pytorch_model_analysis
2. **正确的函数签名**：根据模型参数确定Triton函数需要的所有参数
3. **数学等价性**：确保Triton实现与PyTorch Model.forward()完全等价
4. **正确的4D张量处理**：对于Conv2D等操作，正确处理batch维度
5. **参数使用一致性**：使用与PyTorch模型相同的权重和偏置参数
6. **边界检查和索引**：确保所有内存访问都有正确的边界检查
7. **简洁高效**：优先保证正确性，然后考虑性能优化

## Conv2D实现指导（如果适用）
- **理解操作序列**：Conv2D + ReLU + BiasAdd是三个独立操作
- **参数区分**：区分conv内置bias和额外bias参数
- **索引计算**：正确计算4D张量的batch, channel, height, width索引
- **内存布局**：考虑NCHW格式的内存访问模式

请提供以下结果（JSON格式）:

```json
{{
    "kernel_code": "完整的Triton实现代码，只包含kernel函数和wrapper函数，不包含测试函数",
    "kernel_name": "kernel函数名称",
    "wrapper_name": "wrapper函数名称",
    "launch_config": {{
        "grid_function": "网格配置函数代码",
        "block_sizes": "推荐的块大小配置"
    }},
    "usage_example": "wrapper函数使用示例代码",
    "optimization_notes": ["优化说明1", "优化说明2"]
}}
```

**重要提醒：**
- 不要生成任何测试函数或PyTorch参考实现
- 测试由KernelGen系统使用KernelBench原生模型自动完成
- 只需要生成kernel函数和wrapper函数

注意：
- **只生成kernel函数和wrapper函数，不要生成测试函数**
- kernel代码必须包含所有必要的import语句（import torch, import triton, import triton.language as tl）
- 使用标准的Triton kernel模式：pid = tl.program_id(axis=0), offsets = block_start + tl.arange(0, BLOCK_SIZE)
- wrapper函数必须能直接调用，接受torch.Tensor参数并返回torch.Tensor结果
- **测试由KernelGen系统自动完成，使用KernelBench原生模型**
- 确保正确的索引计算和边界检查
- 保持代码简洁，优先保证正确性而非复杂优化"""
)

# 修复生成提示模板
FIX_GENERATION_PROMPT = PromptTemplate(
    input_variables=["pytorch_code", "current_code", "error_info", "fix_guidance"],
    template="""你是一个专业的Triton kernel调试专家。请根据错误信息和修复指导来修复Triton kernel。

## PyTorch参考代码
```python
{pytorch_code}
```

## 当前有问题的Triton代码
```python
{current_code}
```

## 错误信息
{error_info}

## 修复指导
{fix_guidance}

## Triton语法约束（必须遵守）
1. **网格维度限制**: 最多支持3维网格，program_id(axis)中axis只能是0,1,2
2. **Mask类型匹配**: tl.load()中mask参数必须与pointer参数的维度匹配
3. **索引计算**: 所有索引必须是标量或正确维度的张量
4. **内存访问**: 指针运算必须正确，避免越界访问

## 修复要求
1. **严格遵守Triton语法约束**，特别是网格维度和mask类型限制
2. 修复所有识别出的编译错误
3. 简化复杂的索引计算，确保维度匹配
4. 包含所有必要的导入语句（import torch, import triton等）
5. 保持kernel的核心功能不变

请提供以下修复结果（JSON格式）:

```json
{{
    "fixed_kernel_code": "修复后的完整Triton实现代码，包含kernel函数和wrapper函数",
    "kernel_name": "kernel函数名称",
    "wrapper_name": "wrapper函数名称",
    "changes_made": ["修改1", "修改2", "修改3"],
    "fix_reasoning": {{
        "error_fixes": "错误修复说明",
        "performance_improvements": "性能改进说明",
        "stability_enhancements": "稳定性增强说明"
    }},
    "launch_config": {{
        "grid_function": "更新的网格配置函数代码",
        "block_sizes": "调整后的块大小配置"
    }}
}}
```

注意：
- 必须修复所有报告的错误
- 包含所有必要的导入语句（import torch, import triton, import triton.language as tl）
- 保持代码的可读性和维护性
- 确保边界条件处理正确"""
)

# 智能修复提示模板
INTELLIGENT_FIX_PROMPT = PromptTemplate(
    input_variables=["pytorch_code", "current_code", "error_analysis"],
    template="""基于硬件特性和专业知识修复Triton kernel代码。

## PyTorch参考代码
```python
{pytorch_code}
```

## 当前问题代码
```python
{current_code}
```

## 错误分析（包含硬件信息和知识库）
{error_analysis}

基于硬件特性和操作类型知识，请提供优化的修复方案（JSON格式）：
```json
{{
    "fixed_kernel_code": "修复后的完整代码",
    "changes_made": ["修改1", "修改2", "..."],
    "fix_reasoning": "修复说明",
    "hardware_optimizations": ["硬件优化1", "硬件优化2", "..."],
    "performance_improvements": "预期性能改进"
}}
```"""
)

def get_generation_prompt(is_initial: bool = True) -> PromptTemplate:
    """获取生成提示模板"""
    return INITIAL_GENERATION_PROMPT if is_initial else FIX_GENERATION_PROMPT

def get_intelligent_fix_prompt() -> PromptTemplate:
    """获取智能修复提示模板"""
    return INTELLIGENT_FIX_PROMPT