"""
验证阶段的LangChain提示模板
"""

from langchain_core.prompts import PromptTemplate

# 验证分析提示模板
VALIDATION_PROMPT = PromptTemplate(
    input_variables=["pytorch_code", "triton_code", "test_results", "error_info"],
    template="""你是一个专业的GPU kernel验证专家。请分析Triton kernel的测试结果并提供验证报告。

## PyTorch参考代码
```python
{pytorch_code}
```

## Triton kernel代码
```python
{triton_code}
```

## 测试结果
{test_results}

## 错误信息（如有）
{error_info}

## 验证分析要求
请提供以下验证分析（JSON格式）:

```json
{{
    "validation_result": {{
        "status": "success/partial_success/failure",
        "correctness_score": "正确性评分(0-1)",
        "performance_score": "性能评分(0-1)",
        "overall_quality": "整体质量评估"
    }},
    "detailed_analysis": {{
        "correctness_analysis": {{
            "passes_correctness": "是否通过正确性测试",
            "numerical_accuracy": "数值精度分析",
            "edge_cases": "边界情况处理"
        }},
        "performance_analysis": {{
            "speedup_achieved": "实际加速比",
            "memory_efficiency": "内存效率评估",
            "optimization_effectiveness": "优化效果"
        }},
        "code_quality": {{
            "readability": "代码可读性",
            "maintainability": "可维护性",
            "triton_best_practices": "Triton最佳实践遵循"
        }}
    }},
    "recommendations": {{
        "next_iteration_focus": ["下次迭代重点1", "下次迭代重点2"],
        "optimization_opportunities": ["优化机会1", "优化机会2"],
        "potential_issues": ["潜在问题1", "潜在问题2"]
    }},
    "feedback_for_generator": {{
        "positive_aspects": ["做得好的方面"],
        "areas_for_improvement": ["需要改进的方面"],
        "specific_suggestions": ["具体建议"]
    }}
}}
```

注意：
- 基于实际测试数据进行客观分析
- 提供建设性的改进建议
- 考虑正确性、性能和代码质量的平衡"""
)

# 性能优化分析提示模板
PERFORMANCE_OPTIMIZATION_PROMPT = PromptTemplate(
    input_variables=["current_code", "performance_metrics", "target_performance"],
    template="""你是一个专业的Triton kernel性能优化专家。请分析当前kernel的性能表现并提供优化建议。

## 当前Triton Kernel代码
```python
{current_code}
```

## 性能测试结果
{performance_metrics}

## 目标性能指标
{target_performance}

## 性能优化分析要求

请基于性能测试数据进行深入分析，并提供具体的优化建议（JSON格式）：

```json
{{
    "performance_analysis": {{
        "current_performance": {{
            "speedup_ratio": "当前加速比",
            "execution_time_ms": "执行时间(毫秒)",
            "performance_rating": "性能评级(excellent/good/fair/poor)"
        }},
        "bottleneck_analysis": {{
            "primary_bottlenecks": ["主要性能瓶颈1", "主要性能瓶颈2"],
            "memory_access_pattern": "内存访问模式分析",
            "compute_efficiency": "计算效率分析",
            "parallelization_issues": "并行化问题分析"
        }},
        "gap_analysis": {{
            "target_vs_current": "目标与当前性能差距",
            "improvement_potential": "改进潜力评估",
            "feasibility_assessment": "优化可行性评估"
        }}
    }},
    "optimization_recommendations": {{
        "high_priority": [
            {{
                "optimization": "高优先级优化1",
                "description": "详细描述",
                "expected_impact": "预期影响",
                "implementation_complexity": "实现复杂度(low/medium/high)"
            }}
        ],
        "medium_priority": [
            {{
                "optimization": "中优先级优化1", 
                "description": "详细描述",
                "expected_impact": "预期影响",
                "implementation_complexity": "实现复杂度"
            }}
        ],
        "algorithmic_improvements": [
            "算法层面改进建议1",
            "算法层面改进建议2"
        ],
        "triton_specific_optimizations": [
            "Triton特定优化建议1",
            "Triton特定优化建议2"
        ]
    }},
    "next_iteration_guidance": {{
        "focus_areas": ["下次迭代重点关注领域1", "下次迭代重点关注领域2"],
        "code_changes_needed": [
            {{
                "change_type": "修改类型(memory_access/compute_pattern/grid_config等)",
                "description": "具体修改描述",
                "priority": "优先级(high/medium/low)"
            }}
        ],
        "performance_targets": {{
            "realistic_speedup": "现实可达成的加速比目标",
            "stretch_goal": "挑战性目标",
            "key_metrics_to_improve": ["需要改进的关键指标"]
        }}
    }},
    "technical_insights": {{
        "memory_bandwidth_utilization": "内存带宽利用率分析",
        "compute_intensity": "计算密度分析", 
        "cache_efficiency": "缓存效率分析",
        "thread_divergence": "线程分歧分析",
        "occupancy_analysis": "占用率分析"
    }},
    "code_quality_assessment": {{
        "triton_best_practices": "Triton最佳实践遵循情况",
        "readability_maintainability": "代码可读性和可维护性",
        "optimization_opportunities": "进一步优化机会"
    }}
}}
```

## 分析指导原则

### 🎯 性能瓶颈识别
1. **内存访问模式**：分析是否存在非合并访问、缓存失效等问题
2. **计算与内存比例**：评估是否为内存绑定或计算绑定
3. **并行效率**：检查线程利用率和负载均衡
4. **网格配置**：评估block size和grid dimension的合理性

### 🚀 优化策略建议
1. **向量化**：利用Triton的向量化能力
2. **内存合并**：优化内存访问模式
3. **共享内存**：合理使用shared memory减少全局内存访问
4. **循环展开**：减少循环开销
5. **数据重用**：提高数据局部性

### 📊 性能目标设定
- 基于硬件理论峰值设定现实目标
- 考虑算法复杂度的理论限制
- 参考同类优化kernel的性能表现

### ⚡ Triton特定优化
1. **Block tiling**：合理的数据分块策略
2. **Mask优化**：高效的边界处理
3. **Load/Store优化**：批量内存操作
4. **Constexpr使用**：编译时常量优化

注意：
- 基于实际性能数据进行客观分析
- 提供可操作的具体建议
- 考虑优化的实现复杂度和收益比
- 针对具体的算子类型给出专业建议"""
)

def get_validation_prompt() -> PromptTemplate:
    """获取验证提示模板"""
    return VALIDATION_PROMPT

def get_performance_optimization_prompt() -> PromptTemplate:
    """获取性能优化分析提示模板"""
    return PERFORMANCE_OPTIMIZATION_PROMPT