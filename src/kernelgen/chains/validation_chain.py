"""
验证链 - 使用LangChain实现的代码验证功能
"""

import json
import logging
from typing import Dict, Any

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from ..prompts.validation_prompts import get_validation_prompt

logger = logging.getLogger(__name__)

class ValidationChain:
    """验证链 - 验证和分析Triton kernel"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = get_validation_prompt()
        self.chain = self.prompt | llm | StrOutputParser()
    
    async def validate_kernel(self, pytorch_code: str, triton_code: str,
                            test_results: Dict[str, Any], error_info: str = "",
                            performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        验证Triton kernel
        
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
            # 准备输入数据
            input_data = {
                "pytorch_code": pytorch_code,
                "triton_code": triton_code,
                "test_results": json.dumps(test_results, indent=2, ensure_ascii=False),
                "error_info": error_info or "无错误"
            }
            
            # 调用LLM
            response = await self.chain.ainvoke(input_data)
            
            # 解析响应
            validation_result = self._parse_validation_response(response, test_results)
            
            return {
                "success": True,
                "result": validation_result
            }
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
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