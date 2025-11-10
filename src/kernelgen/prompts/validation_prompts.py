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

def get_validation_prompt() -> PromptTemplate:
    """获取验证提示模板"""
    return VALIDATION_PROMPT