"""
验证链 - 使用LangChain实现的代码验证功能
"""

import json
import logging
from typing import Dict, Any, Optional, List

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.validation_prompts import get_validation_prompt
from ..services import CodeValidationService

logger = logging.getLogger(__name__)

class ValidationChain:
    """验证链 - 验证和分析Triton kernel"""
    
    def __init__(self, llm: BaseChatModel, 
                 validation_service: Optional[CodeValidationService] = None):
        self.llm = llm
        self.validation_service = validation_service or CodeValidationService()
        self.prompt = get_validation_prompt()
        self.chain = self.prompt | llm | StrOutputParser()
    
    async def validate_kernel(self, pytorch_code: str, triton_code: str,
                            test_results: Dict[str, Any], error_info: str = "",
                            performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        综合验证Triton kernel - 结合规则验证和LLM分析
        
        Args:
            pytorch_code: PyTorch参考代码
            triton_code: Triton kernel代码
            test_results: 测试结果
            error_info: 错误信息
            performance_data: 性能数据
            
        Returns:
            验证结果
        """
        try:
            # 1. 静态代码验证
            static_validation = self.validation_service.validate_triton_code(triton_code)
            
            # 2. LLM验证分析
            llm_validation = await self._llm_validation_analysis(
                pytorch_code, triton_code, test_results, error_info
            )
            
            # 3. 综合验证结果
            combined_result = self._combine_validation_results(
                static_validation, llm_validation, test_results
            )
            
            return {
                "success": True,
                "result": combined_result
            }
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _llm_validation_analysis(self, pytorch_code: str, triton_code: str,
                                     test_results: Dict[str, Any], error_info: str) -> Dict[str, Any]:
        """LLM验证分析"""
        input_data = {
            "pytorch_code": pytorch_code,
            "triton_code": triton_code,
            "test_results": json.dumps(test_results, indent=2, ensure_ascii=False),
            "error_info": error_info or "无错误"
        }
        
        response = await self.chain.ainvoke(input_data)
        return self._parse_validation_response(response, test_results)
    
    def _combine_validation_results(self, static_result, llm_result: Dict[str, Any], 
                                   test_results: Dict[str, Any]) -> Dict[str, Any]:
        """综合静态验证和LLM验证结果"""
        
        # 基于测试结果的基础评估
        correctness = test_results.get("correctness", False)
        speedup = test_results.get("speedup", 0.0)
        
        # 综合质量评分
        static_score = static_result.score
        performance_score = min(speedup / 2.0, 1.0) if speedup > 0 else 0.0
        correctness_score = 1.0 if correctness else 0.0
        
        overall_score = (static_score * 0.3 + performance_score * 0.3 + correctness_score * 0.4)
        
        # 确定状态
        if correctness and static_result.is_valid and speedup >= 1.0:
            status = "success"
        elif correctness and static_result.is_valid:
            status = "partial_success"
        else:
            status = "failure"
        
        return {
            "validation_result": {
                "status": status,
                "correctness_score": correctness_score,
                "performance_score": performance_score,
                "code_quality_score": static_score,
                "overall_score": overall_score
            },
            "static_validation": {
                "is_valid": static_result.is_valid,
                "issues": static_result.issues,
                "warnings": static_result.warnings,
                "suggestions": static_result.suggestions
            },
            "llm_analysis": llm_result.get("detailed_analysis", {}),
            "recommendations": {
                "code_improvements": static_result.suggestions,
                "performance_improvements": llm_result.get("recommendations", {}).get("optimization_opportunities", []),
                "next_iteration_focus": self._determine_next_focus(static_result, test_results)
            },
            "summary": self._generate_validation_summary(status, overall_score, static_result, test_results)
        }
    
    def _determine_next_focus(self, static_result, test_results: Dict[str, Any]) -> List[str]:
        """确定下次迭代的重点"""
        focus_areas = []
        
        if not test_results.get("correctness", False):
            focus_areas.append("修复正确性问题")
        
        if static_result.issues:
            focus_areas.append("解决代码质量问题")
        
        if test_results.get("speedup", 0) < 1.0:
            focus_areas.append("提升性能表现")
        
        if not focus_areas:
            focus_areas.append("进一步优化性能")
        
        return focus_areas
    
    def _generate_validation_summary(self, status: str, score: float, 
                                   static_result, test_results: Dict[str, Any]) -> str:
        """生成验证摘要"""
        summary = f"验证状态: {status} (综合评分: {score:.2f})\n"
        
        if test_results.get("correctness"):
            summary += "✅ 正确性验证通过\n"
        else:
            summary += "❌ 正确性验证失败\n"
        
        if static_result.is_valid:
            summary += "✅ 代码质量验证通过\n"
        else:
            summary += f"❌ 发现 {len(static_result.issues)} 个代码问题\n"
        
        speedup = test_results.get("speedup", 0.0)
        if speedup >= 1.0:
            summary += f"✅ 性能表现良好 (加速比: {speedup:.2f}x)\n"
        else:
            summary += f"⚠️ 性能有待提升 (加速比: {speedup:.2f}x)\n"
        
        return summary
    
    async def analyze_performance_optimization(self, current_code: str, 
                                             performance_metrics: Dict[str, Any],
                                             target_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析性能优化机会
        
        Args:
            current_code: 当前代码
            performance_metrics: 性能指标
            target_performance: 目标性能
            
        Returns:
            性能优化分析结果
        """
        try:
            # 使用性能优化分析提示
            perf_optimization_prompt = get_performance_optimization_prompt()
            perf_optimization_chain = perf_optimization_prompt | self.llm | StrOutputParser()
            
            # 准备输入数据
            input_data = {
                "current_code": current_code,
                "performance_metrics": json.dumps(performance_metrics, indent=2, ensure_ascii=False),
                "target_performance": json.dumps(target_performance or {"target_speedup": "2.0x"}, indent=2, ensure_ascii=False)
            }
            
            # 调用LLM进行性能分析
            response = await perf_optimization_chain.ainvoke(input_data)
            
            # 解析分析结果
            optimization_result = self._parse_validation_response(response, performance_metrics)
            optimization_result["analysis_type"] = "performance_optimization"
            
            logger.info("完成性能优化分析")
            return {
                "success": True,
                "result": optimization_result
            }
            
        except Exception as e:
            logger.error(f"性能优化分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_validation_response(self, response: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """解析验证响应"""
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            import re
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 如果都失败了，基于测试结果生成基本验证结果
            logger.warning("无法解析验证响应，使用基于测试结果的默认结构")
            
            correctness = test_results.get("correctness", False)
            speedup = test_results.get("speedup", 0.0)
            
            # 确定状态
            if correctness and speedup >= 1.0:
                status = "success"
            elif correctness:
                status = "partial_success"
            else:
                status = "failure"
            
            return {
                "validation_result": {
                    "status": status,
                    "correctness_score": 1.0 if correctness else 0.0,
                    "performance_score": min(speedup / 2.0, 1.0) if speedup > 0 else 0.0,
                    "overall_quality": "基于测试结果的自动评估"
                },
                "detailed_analysis": {
                    "correctness_analysis": {
                        "passes_correctness": correctness,
                        "numerical_accuracy": "基于测试结果",
                        "edge_cases": "需要进一步测试"
                    },
                    "performance_analysis": {
                        "speedup_achieved": speedup,
                        "memory_efficiency": "需要分析",
                        "optimization_effectiveness": "基于加速比评估"
                    },
                    "code_quality": {
                        "readability": "需要人工评估",
                        "maintainability": "需要人工评估",
                        "triton_best_practices": "需要检查"
                    }
                },
                "recommendations": {
                    "next_iteration_focus": ["提高正确性" if not correctness else "优化性能"],
                    "optimization_opportunities": ["内存访问优化", "计算优化"],
                    "potential_issues": ["边界条件", "数值精度"]
                },
                "feedback_for_generator": {
                    "positive_aspects": ["生成了可编译的代码"] if test_results.get("success") else [],
                    "areas_for_improvement": ["正确性"] if not correctness else ["性能优化"],
                    "specific_suggestions": ["检查算法实现", "优化内存访问模式"]
                },
                "raw_response": response
            }