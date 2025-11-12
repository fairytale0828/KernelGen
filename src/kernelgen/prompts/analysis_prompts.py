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

## 重要说明
这个PyTorch代码包含：
1. `get_inputs()`: 返回测试输入张量列表，这些是forward函数的参数
2. `get_init_inputs()`: 返回模型初始化参数，用于创建Model实例
3. `Model`类: 需要用get_init_inputs()的返回值初始化
4. 你需要分析Model.forward()方法的计算逻辑来设计Triton kernel

## 关键分析要求
**必须逐步分解PyTorch模型的每个操作：**
1. **仔细分析Model.__init__()**: 识别所有层和参数（self.自带的模型层和参数）
2. **逐行分析Model.forward()**: 理解每一步的具体计算
3. **识别复合操作**: 区分单一操作vs复合操作序列
4. **参数依赖分析**: 确定Triton kernel需要哪些输入参数
5. **数据流分析**: 理解张量形状在每步如何变化

## 特别注意
- 如果使用nn.Conv2d，注意识别他是否包含bias参数
- 如果有额外的bias操作，要区分conv内置bias和额外bias
- 必须确保Triton实现与PyTorch模型在数学上完全等价

## 分析要求
请提供以下分析结果（JSON格式）:

```json
{{
    "pytorch_model_analysis": {{
        "model_layers": ["层1描述", "层2描述", "..."],
        "forward_steps": ["步骤1: 具体操作", "步骤2: 具体操作", "..."],
        "parameters_needed": ["参数1", "参数2", "..."],
        "operation_sequence": "完整的操作序列描述"
    }},
    "operation_type": "操作类型(如: conv2d_composite, elementwise, reduction, matmul等)",
    "computational_pattern": "计算模式描述",
    "memory_access_pattern": "内存访问模式",
    "parallelization_strategy": "并行化策略",
    "triton_kernel_design": {{
        "kernel_parameters": ["kernel需要的参数列表"],
        "algorithm_approach": "算法实现方法(如: direct_conv, im2col等)",
        "indexing_strategy": "4D张量索引策略",
        "batch_handling": "batch维度处理方式"
    }},
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
        "key_optimizations": ["关键优化点1", "关键优化点2", "..."],
        "potential_challenges": ["潜在挑战1", "潜在挑战2", "..."],
        "triton_features": ["需要使用的Triton特性"],
        "correctness_requirements": ["正确性要求1", "正确性要求2", "..."]
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

# 错误分析提示模板
ERROR_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["error_info", "code_context", "performance_data"],
    template="""分析Triton kernel错误并提供修复方案。

## 错误信息
{error_info}

## 代码上下文
```python
{code_context}
```

## 性能数据
{performance_data}

请分析错误原因并提供修复建议（JSON格式）：
```json
{{
    "error_type": "错误类型",
    "root_cause": "根本原因",
    "fix_strategy": ["修复方案1", "修复方案2"],
    "code_changes": ["需要修改的代码部分1", "需要修改的代码部分2"]
}}
```"""
)

def get_analysis_prompt(is_initial: bool = True) -> PromptTemplate:
    """获取分析提示模板"""
    return INITIAL_ANALYSIS_PROMPT if is_initial else DEBUG_ANALYSIS_PROMPT

# 性能优化分析提示模板
PERFORMANCE_OPTIMIZATION_PROMPT = PromptTemplate(
    input_variables=["current_code", "performance_metrics", "target_performance"],
    template="""分析Triton kernel性能并提供优化建议。

## 当前代码
```python
{current_code}
```

## 性能指标
{performance_metrics}

## 目标性能
{target_performance}

请分析性能瓶颈并提供优化方案（JSON格式）：
```json
{{
    "bottlenecks": ["瓶颈1", "瓶颈2"],
    "optimizations": ["优化建议1", "优化建议2"],
    "expected_improvement": "预期提升"
}}
```"""
)

def get_error_analysis_prompt() -> PromptTemplate:
    """获取错误分析提示模板"""
    return ERROR_ANALYSIS_PROMPT

def get_hardware_context(device_info: dict) -> str:
    """生成硬件上下文信息"""
    return f"""
## GPU硬件信息
- 设备型号: {device_info.get('name', 'Unknown')}
- 计算能力: {device_info.get('compute_capability', 'Unknown')}
- 内存大小: {device_info.get('memory_size', 'Unknown')}
- SM数量: {device_info.get('sm_count', 'Unknown')}

## 硬件优化建议
- 内存合并访问: 确保连续内存访问模式
- 共享内存利用: 充分利用片上高速缓存
- 线程束效率: 避免分支分歧，保持32线程束同步
- 寄存器使用: 平衡寄存器使用和SM占用率
"""

def get_knowledge_context(operation_type: str, knowledge_base: dict = None) -> str:
    """生成知识库上下文信息"""
    if not knowledge_base:
        return "## 知识库\n暂无相关知识库信息"
    
    relevant_knowledge = knowledge_base.get(operation_type, {})
    
    context = f"## {operation_type.upper()}操作相关知识\n"
    
    if relevant_knowledge.get('best_practices'):
        context += "### 最佳实践\n"
        for practice in relevant_knowledge['best_practices']:
            context += f"- {practice}\n"
    
    if relevant_knowledge.get('optimization_tips'):
        context += "### 优化技巧\n"
        for tip in relevant_knowledge['optimization_tips']:
            context += f"- {tip}\n"
    
    if relevant_knowledge.get('common_issues'):
        context += "### 常见问题\n"
        for issue in relevant_knowledge['common_issues']:
            context += f"- {issue}\n"
    
    return context

def get_performance_optimization_prompt() -> PromptTemplate:
    """获取性能优化分析提示模板"""
    return PERFORMANCE_OPTIMIZATION_PROMPT