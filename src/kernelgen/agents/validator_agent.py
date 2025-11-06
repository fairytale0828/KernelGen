"""
ValidatorAgent - 验证和性能分析专家
基于aikg的Conductor模式，负责测试运行、错误分析和性能评估
"""

import json
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ValidatorAgent(BaseAgent):
    """
    验证和性能分析专家Agent (基于aikg的Conductor模式)
    
    职责：
    1. 执行Triton kernel的编译和运行测试
    2. 收集和分析错误信息
    3. 评估性能指标和加速比
    4. 提供反馈和优化建议
    5. 决定是否继续迭代
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("ValidatorAgent", llm_client, config)
        
        # 加载prompt模板
        self._load_prompt_templates()
        
        logger.info("ValidatorAgent初始化完成")
    
    def _load_prompt_templates(self):
        """加载prompt模板"""
        self.validation_analysis_template = """你是一个专业的GPU kernel验证和性能分析专家。

## 任务：分析Triton kernel的测试结果并提供反馈

### PyTorch参考代码：
```python
{pytorch_code}
```

### 生成的Triton代码：
```python
{triton_code}
```

### 测试结果：
{test_results}

### 错误信息（如有）：
{error_info}

### 性能数据（如有）：
{performance_data}

### 分析要求：

1. **正确性分析**：
   - 评估代码是否正确编译和运行
   - 分析输出结果的正确性
   - 识别潜在的数值稳定性问题

2. **性能分析**：
   - 评估加速比和性能表现
   - 识别性能瓶颈
   - 分析内存访问效率

3. **错误诊断**：
   - 分析编译错误的根本原因
   - 诊断运行时错误
   - 提供具体的修复建议

4. **优化建议**：
   - 提供性能优化方向
   - 建议架构调整
   - 推荐参数调优

5. **迭代决策**：
   - 判断是否需要继续迭代
   - 评估当前结果的质量
   - 提供下一步行动建议

请按照以下JSON格式输出分析结果：

```json
{{
  "validation_result": {{
    "status": "success|error|suboptimal",
    "correctness_score": 0.0-1.0,
    "performance_score": 0.0-1.0,
    "overall_assessment": "整体评估"
  }},
  "error_analysis": {{
    "has_errors": true|false,
    "error_type": "compile|runtime|correctness|performance",
    "root_cause": "错误根本原因",
    "severity": "low|medium|high|critical",
    "fix_suggestions": ["修复建议1", "修复建议2"]
  }},
  "performance_analysis": {{
    "speedup_achieved": 0.0,
    "efficiency_score": 0.0-1.0,
    "bottleneck_type": "memory|compute|synchronization|none",
    "optimization_opportunities": ["优化机会1", "优化机会2"]
  }},
  "feedback_for_analyzer": {{
    "architecture_issues": ["架构问题1", "架构问题2"],
    "design_suggestions": ["设计建议1", "设计建议2"],
    "parameter_adjustments": {{"param1": "value1", "param2": "value2"}}
  }},
  "feedback_for_generator": {{
    "code_issues": ["代码问题1", "代码问题2"],
    "implementation_fixes": ["实现修复1", "实现修复2"],
    "optimization_hints": ["优化提示1", "优化提示2"]
  }},
  "iteration_decision": {{
    "should_continue": true|false,
    "reason": "继续/停止的原因",
    "next_focus": "下一轮重点关注的方面",
    "confidence_level": 0.0-1.0
  }}
}}
```"""
    
    async def validate_kernel(self, 
                            pytorch_code: str,
                            triton_code: str,
                            test_results: Dict[str, Any],
                            error_info: str = "",
                            performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        验证kernel并提供反馈
        
        Args:
            pytorch_code: PyTorch参考代码
            triton_code: 生成的Triton代码
            test_results: 测试结果
            error_info: 错误信息
            performance_data: 性能数据
            
        Returns:
            验证结果字典
        """
        try:
            input_data = {
                "pytorch_code": pytorch_code,
                "triton_code": triton_code,
                "test_results": self._format_test_results(test_results),
                "error_info": error_info or "无错误信息",
                "performance_data": self._format_performance_data(performance_data or {})
            }
            
            prompt = self.validation_analysis_template.format(**input_data)
            
            response = self.generate_llm_response(prompt)
            
            # 解析JSON响应
            result = self._extract_json_from_response(response)
            if result:
                logger.info("ValidatorAgent验证分析完成")
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
            logger.error(f"ValidatorAgent执行失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_test_results(self, test_results: Dict[str, Any]) -> str:
        """格式化测试结果"""
        if not test_results:
            return "无测试结果"
        
        formatted = []
        for key, value in test_results.items():
            if key == "success":
                formatted.append(f"**测试状态**: {'成功' if value else '失败'}")
            elif key == "correctness":
                formatted.append(f"**正确性**: {'通过' if value else '未通过'}")
            elif key == "speedup":
                formatted.append(f"**加速比**: {value:.2f}x")
            elif key == "error":
                formatted.append(f"**错误**: {value}")
            else:
                formatted.append(f"**{key}**: {value}")
        
        return "\n".join(formatted)
    
    def _format_performance_data(self, performance_data: Dict[str, Any]) -> str:
        """格式化性能数据"""
        if not performance_data:
            return "无性能数据"
        
        formatted = []
        for key, value in performance_data.items():
            if "time" in key.lower():
                if isinstance(value, (int, float)):
                    formatted.append(f"**{key}**: {value:.4f}ms")
                else:
                    formatted.append(f"**{key}**: {value}")
            elif "speedup" in key.lower():
                if isinstance(value, (int, float)):
                    formatted.append(f"**{key}**: {value:.2f}x")
                else:
                    formatted.append(f"**{key}**: {value}")
            else:
                formatted.append(f"**{key}**: {value}")
        
        return "\n".join(formatted)
    
    def extract_feedback_for_analyzer(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取给AnalyzerAgent的反馈"""
        if not validation_result or "result" not in validation_result:
            return {}
        
        result = validation_result["result"]
        feedback = {}
        
        # 提取架构相关的反馈
        if "feedback_for_analyzer" in result:
            feedback.update(result["feedback_for_analyzer"])
        
        # 添加错误分析信息
        if "error_analysis" in result:
            error_analysis = result["error_analysis"]
            feedback["error_info"] = {
                "error_type": error_analysis.get("error_type", ""),
                "root_cause": error_analysis.get("root_cause", ""),
                "severity": error_analysis.get("severity", ""),
                "fix_suggestions": error_analysis.get("fix_suggestions", [])
            }
        
        # 添加性能分析信息
        if "performance_analysis" in result:
            perf_analysis = result["performance_analysis"]
            feedback["performance_info"] = {
                "speedup_achieved": perf_analysis.get("speedup_achieved", 0.0),
                "bottleneck_type": perf_analysis.get("bottleneck_type", ""),
                "optimization_opportunities": perf_analysis.get("optimization_opportunities", [])
            }
        
        return feedback
    
    def extract_feedback_for_generator(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取给GeneratorAgent的反馈"""
        if not validation_result or "result" not in validation_result:
            return {}
        
        result = validation_result["result"]
        feedback = {}
        
        # 提取代码相关的反馈
        if "feedback_for_generator" in result:
            feedback.update(result["feedback_for_generator"])
        
        # 添加错误分析信息
        if "error_analysis" in result:
            error_analysis = result["error_analysis"]
            feedback["error_info"] = {
                "error_type": error_analysis.get("error_type", ""),
                "root_cause": error_analysis.get("root_cause", ""),
                "fix_suggestions": error_analysis.get("fix_suggestions", [])
            }
        
        return feedback
    
    def should_continue_iteration(self, validation_result: Dict[str, Any]) -> tuple[bool, str]:
        """判断是否应该继续迭代"""
        if not validation_result or "result" not in validation_result:
            return True, "验证结果不完整，需要继续迭代"
        
        result = validation_result["result"]
        
        # 检查迭代决策
        if "iteration_decision" in result:
            decision = result["iteration_decision"]
            should_continue = decision.get("should_continue", True)
            reason = decision.get("reason", "未提供原因")
            return should_continue, reason
        
        # 基于验证结果做决策
        if "validation_result" in result:
            validation = result["validation_result"]
            status = validation.get("status", "error")
            
            if status == "success":
                return False, "验证成功，无需继续迭代"
            elif status == "error":
                return True, "存在错误，需要继续迭代修复"
            else:  # suboptimal
                return True, "性能不佳，需要继续优化"
        
        return True, "默认继续迭代"
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Agent的主要执行方法，兼容BaseAgent的抽象方法
        
        Args:
            **kwargs: 关键字参数
            
        Returns:
            验证结果字典
        """
        pytorch_code = kwargs.get("pytorch_code", "")
        triton_code = kwargs.get("triton_code", "")
        test_results = kwargs.get("test_results", {})
        error_info = kwargs.get("error_info", "")
        performance_data = kwargs.get("performance_data", {})
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.validate_kernel(pytorch_code, triton_code, test_results, error_info, performance_data)
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