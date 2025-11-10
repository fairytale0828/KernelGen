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

## PyTorch代码
```python
{pytorch_code}
```

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

## 错误信息
{error_info}

## 性能信息
{performance_info}

## 调试分析要求
请提供以下调试分析（JSON格式）:

```json
{{
    "error_analysis": {{
        "error_type": "错误类型(语法/编译/运行时/正确性/性能)",
        "root_cause": "根本原因分析",
        "affected_components": ["受影响的代码组件"]
    }},
    "fix_strategy": {{
        "primary_fixes": ["主要修复方案1", "主要修复方案2"],
        "code_changes": {{
            "kernel_signature": "是否需要修改kernel签名",
            "block_size": "是否需要调整块大小",
            "memory_access": "是否需要优化内存访问",
            "computation_logic": "是否需要修改计算逻辑"
        }}
    }},
    "updated_guidance": {{
        "new_optimizations": ["新的优化建议"],
        "avoid_patterns": ["需要避免的模式"],
        "triton_best_practices": ["Triton最佳实践"]
    }}
}}
```"""
)

def get_analysis_prompt(is_initial: bool = True) -> PromptTemplate:
    """获取分析提示模板"""
    return INITIAL_ANALYSIS_PROMPT if is_initial else DEBUG_ANALYSIS_PROMPT