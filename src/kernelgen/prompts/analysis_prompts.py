"""
分析阶段的LangChain提示模板
"""

from langchain_core.prompts import PromptTemplate

# 初始分析提示模板
INITIAL_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["pytorch_code", "operation_name", "input_shapes", "output_shapes"],
    template="""你是一个专业的GPU kernel分析专家。请分析以下PyTorch操作并设计Triton kernel架构。

## 操作信息
- 操作名称: {operation_name}
- 输入形状: {input_shapes}
- 输出形状: {output_shapes}

## PyTorch代码（来自KernelBench数据集）
```python
{pytorch_code}
```

## 重要说明
这个PyTorch代码包含：
1. `get_inputs()`: 返回测试输入张量列表，这些是forward函数的参数
2. `get_init_inputs()`: 返回模型初始化参数，用于创建Model实例
3. `Model`类: 需要用get_init_inputs()的返回值初始化
4. 你需要分析Model.forward()方法的计算逻辑来设计Triton kernel

## 分析要求
请提供以下分析结果（JSON格式）:

```json
{{
    "operation_type": "操作类型(如: elementwise, reduction, matmul等)",
    "computational_pattern": "计算模式描述",
    "memory_access_pattern": "内存访问模式",
    "parallelization_strategy": "并行化策略",
    "block_size_recommendations": {{
        "BLOCK_SIZE": "推荐的块大小",
        "reasoning": "选择理由"
    }},
    "kernel_structure": {{
        "grid_dimensions": "网格维度设计",
        "thread_organization": "线程组织方式",
        "memory_hierarchy": "内存层次使用"
    }},
    "implementation_guidance": {{
        "key_optimizations": ["关键优化点1", "关键优化点2"],
        "potential_challenges": ["潜在挑战1", "潜在挑战2"],
        "triton_features": ["需要使用的Triton特性"]
    }}
}}
```"""
)

# 调试分析提示模板
DEBUG_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["pytorch_code", "previous_code", "error_info", "performance_info"],
    template="""你是一个专业的GPU kernel调试专家。请分析之前生成的Triton kernel的问题并提供修复方案。

## PyTorch参考代码
```python
{pytorch_code}
```

## 之前生成的Triton代码
```python
{previous_code}
```

## 错误信息（结构化）
{error_info}

## 错误类型说明
- TRITON_GRID_ERROR: 网格维度超限，需要重新设计为3维或更少
- TRITON_MASK_ERROR: mask类型不匹配，需要修复tl.load()调用
- SHAPE_MISMATCH_ERROR: 张量形状不匹配，需要检查索引计算
- COMPILATION_ERROR: 编译错误，需要检查Triton语法
- CORRECTNESS_ERROR: 输出不正确，需要检查算法逻辑
- PERFORMANCE_ERROR: 性能问题，需要优化实现

## 性能信息
{performance_info}

## 常见Triton错误和解决方案
1. **"Mask argument cannot be block type if pointer argument is not a block"**
   - 原因: tl.load()中mask参数类型不匹配
   - 解决: 确保pointer和mask的维度匹配，或使用标量mask

2. **"program_id axis must be 0, 1, or 2 but got X"**
   - 原因: Triton只支持最多3维网格 (axis=0,1,2)
   - 解决: 重新设计网格布局，将多维度合并到3维内

3. **形状不匹配错误**
   - 原因: 张量维度计算错误
   - 解决: 仔细检查索引计算和边界条件

## 调试分析要求
请仔细分析错误信息，特别关注Triton语法限制，提供以下调试分析（JSON格式）:

```json
{{
    "error_analysis": {{
        "error_type": "错误类型(triton_syntax/triton_compilation/shape_mismatch/correctness/performance)",
        "root_cause": "根本原因分析，特别说明违反了哪个Triton限制",
        "affected_components": ["受影响的代码组件"],
        "triton_constraint_violated": "违反的Triton约束（如网格维度限制、mask类型限制等）"
    }},
    "fix_strategy": {{
        "primary_fixes": ["主要修复方案1", "主要修复方案2"],
        "code_changes": {{
            "grid_design": "是否需要重新设计网格布局",
            "mask_handling": "是否需要修复mask使用方式", 
            "indexing_logic": "是否需要修复索引计算",
            "memory_access": "是否需要优化内存访问模式"
        }},
        "specific_triton_fixes": ["针对Triton语法的具体修复建议"]
    }},
    "updated_guidance": {{
        "triton_constraints": ["必须遵守的Triton约束"],
        "avoid_patterns": ["需要避免的错误模式"],
        "recommended_approach": "推荐的实现方法"
    }}
}}
```"""
)

def get_analysis_prompt(is_initial: bool = True) -> PromptTemplate:
    """获取分析提示模板"""
    return INITIAL_ANALYSIS_PROMPT if is_initial else DEBUG_ANALYSIS_PROMPT