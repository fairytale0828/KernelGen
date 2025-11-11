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
- 如果PyTorch模型包含可学习参数（如Conv2d的权重、偏置），Triton wrapper函数应该接受这些参数作为输入
- 例如：如果Model有conv.weight和bias参数，Triton函数应该是 `triton_func(x, weight, bias)`
- 这样可以测试Triton kernel的完整功能，性能测试工具会自动从PyTorch模型中提取这些参数

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

def test_example_correctness():
    \"\"\"测试函数 - 验证Triton实现与PyTorch的等价性\"\"\"
    print("Testing Triton implementation vs PyTorch...")
    
    # 生成测试数据
    x = torch.randn(1024, device='cuda')
    
    # PyTorch参考结果
    pytorch_result = torch.example_operation(x)  # 替换为实际操作
    
    # Triton结果
    triton_result = triton_example(x)
    
    # 验证等价性
    max_diff = torch.max(torch.abs(pytorch_result - triton_result))
    print(f"Maximum difference: {{max_diff.item()}}")
    
    if torch.allclose(pytorch_result, triton_result, rtol=1e-5, atol=1e-8):
        print("✓ Triton implementation matches PyTorch!")
        return True
    else:
        print("✗ Triton implementation differs from PyTorch!")
        return False
```

## 生成要求
1. 生成完整的Triton实现，包含kernel函数、wrapper函数和测试函数
2. 包含所有必要的导入语句（import torch, import triton等）
3. 使用标准的Triton kernel模式：program_id, arange, load, store
4. 包含适当的边界检查（mask）
5. wrapper函数必须能直接接受torch.Tensor参数并返回结果
6. 测试函数必须验证与PyTorch实现的等价性
7. 确保索引计算正确，保持代码简洁

请提供以下结果（JSON格式）:

```json
{{
    "kernel_code": "完整的Triton实现代码，包含kernel函数、wrapper函数和测试函数",
    "kernel_name": "kernel函数名称",
    "wrapper_name": "wrapper函数名称",
    "test_function_name": "测试函数名称",
    "launch_config": {{
        "grid_function": "网格配置函数代码",
        "block_sizes": "推荐的块大小配置"
    }},
    "usage_example": "wrapper函数使用示例代码",
    "optimization_notes": ["优化说明1", "优化说明2"]
}}
```

注意：
- 必须生成完整的实现，包含kernel函数、wrapper函数和测试函数
- kernel代码必须包含所有必要的import语句（import torch, import triton, import triton.language as tl）
- 使用标准的Triton kernel模式：pid = tl.program_id(axis=0), offsets = block_start + tl.arange(0, BLOCK_SIZE)
- wrapper函数必须能直接调用，接受torch.Tensor参数并返回torch.Tensor结果
- 测试函数必须包含完整的正确性验证逻辑，使用torch.allclose进行比较
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

def get_generation_prompt(is_initial: bool = True) -> PromptTemplate:
    """获取生成提示模板"""
    return INITIAL_GENERATION_PROMPT if is_initial else FIX_GENERATION_PROMPT