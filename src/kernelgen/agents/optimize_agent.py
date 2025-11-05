"""
Optimize Agent - 性能优化师
负责分析kernel性能并提供具体的优化建议
"""

import logging
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class OptimizeAgent(BaseAgent):
    """
    优化Agent - 负责分析性能瓶颈并提供优化建议
    
    职责：
    1. 分析kernel性能数据和基准测试结果
    2. 识别性能瓶颈和优化机会
    3. 提供具体的优化建议和策略
    4. 评估优化效果和改进方向
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("OptimizeAgent", llm_client, config)
    
    async def run(self, task_info: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        执行性能分析和优化建议生成
        
        Args:
            task_info: 任务信息字典，包含性能数据和代码
            
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        try:
            # 从task_info中获取性能数据
            performance_data = task_info.get("performance_data", {})
            kernel_code = task_info.get("kernel_code", "")
            
            # 分析性能瓶颈
            bottleneck_analysis = self._analyze_performance_bottleneck(performance_data, kernel_code)
            
            # 生成优化建议
            optimization_suggestions = await self._generate_optimization_suggestions(
                performance_data, kernel_code, bottleneck_analysis
            )
            
            # 构造返回结果
            result_content = {
                "performance_analysis": {
                    "bottleneck_analysis": bottleneck_analysis,
                    "optimization_suggestions": optimization_suggestions,
                    "current_performance": performance_data
                }
            }
            
            # 格式化为JSON字符串
            import json
            formatted_result = json.dumps(result_content, indent=2, ensure_ascii=False)
            
            return formatted_result, "", ""
            
        except Exception as e:
            logger.error(f"性能分析失败: {str(e)}")
            raise
    
    def _analyze_performance_bottleneck(self, performance_data: Dict[str, Any], 
                                      kernel_code: str) -> Dict[str, Any]:
        """分析性能瓶颈"""
        
        speedup = performance_data.get("speedup", 0.0)
        triton_time = performance_data.get("triton_time_ms", 0.0)
        pytorch_time = performance_data.get("pytorch_time_ms", 0.0)
        
        # 基于性能水平推断瓶颈类型
        if speedup < 0.3:
            primary_bottleneck = "algorithm_inefficiency"
            bottleneck_severity = "critical"
        elif speedup < 0.7:
            primary_bottleneck = "memory_bandwidth"
            bottleneck_severity = "high"
        elif speedup < 1.0:
            primary_bottleneck = "compute_utilization"
            bottleneck_severity = "medium"
        else:
            primary_bottleneck = "minor_optimizations"
            bottleneck_severity = "low"
        
        # 代码分析推断瓶颈
        code_issues = self._analyze_code_performance_issues(kernel_code)
        
        return {
            "primary_bottleneck": primary_bottleneck,
            "bottleneck_severity": bottleneck_severity,
            "code_issues": code_issues,
            "performance_level": self._categorize_performance_level(speedup),
            "analysis_confidence": 0.8
        }
    
    def _analyze_code_performance_issues(self, kernel_code: str) -> list:
        """分析代码中的性能问题"""
        
        issues = []
        
        # 检查内存访问模式
        if "tl.load" in kernel_code:
            load_count = kernel_code.count("tl.load")
            if load_count > 5:
                issues.append(f"过多的内存加载操作 ({load_count}次)")
        
        # 检查循环结构
        if "for " in kernel_code:
            issues.append("显式循环可能影响并行性")
        
        # 检查分支语句
        if_count = kernel_code.count("if ")
        if if_count > 3:
            issues.append(f"过多的分支语句 ({if_count}个)")
        
        # 检查同步操作
        if "tl.barrier" in kernel_code:
            issues.append("同步操作可能影响性能")
        
        # 检查BLOCK_SIZE使用
        if "BLOCK_SIZE" not in kernel_code:
            issues.append("未使用BLOCK_SIZE参数化")
        
        return issues
    
    def _categorize_performance_level(self, speedup: float) -> str:
        """分类性能水平"""
        
        if speedup >= 2.0:
            return "excellent"
        elif speedup >= 1.2:
            return "good"
        elif speedup >= 0.8:
            return "acceptable"
        elif speedup >= 0.5:
            return "poor"
        else:
            return "very_poor"
    
    async def _generate_optimization_suggestions(self, performance_data: Dict[str, Any],
                                               kernel_code: str, 
                                               bottleneck_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化建议"""
        
        # 构造优化建议提示
        optimization_prompt = self._create_optimization_prompt(
            performance_data, kernel_code, bottleneck_analysis
        )
        
        # 调用LLM生成优化建议
        llm_response = self.generate_llm_response(
            optimization_prompt,
            "You are a Triton optimization expert. Provide specific, actionable optimization suggestions."
        )
        
        # 解析优化建议
        suggestions = self._parse_optimization_suggestions(llm_response, bottleneck_analysis)
        
        return suggestions
    
    def _create_optimization_prompt(self, performance_data: Dict[str, Any], 
                                  kernel_code: str, bottleneck_analysis: Dict[str, Any]) -> str:
        """创建优化建议提示"""
        
        speedup = performance_data.get("speedup", 0.0)
        triton_time = performance_data.get("triton_time_ms", 0.0)
        pytorch_time = performance_data.get("pytorch_time_ms", 0.0)
        primary_bottleneck = bottleneck_analysis.get("primary_bottleneck", "unknown")
        
        prompt = f"""
为以下Triton kernel提供具体的优化建议：

## 当前性能：
- 加速比: {speedup:.2f}x
- Triton执行时间: {triton_time:.3f}ms
- PyTorch执行时间: {pytorch_time:.3f}ms
- 主要瓶颈: {primary_bottleneck}
- 性能水平: {bottleneck_analysis.get('performance_level', 'unknown')}

## 当前代码：
```python
{kernel_code}
```

## 瓶颈分析：
- 瓶颈严重程度: {bottleneck_analysis.get('bottleneck_severity', 'unknown')}
- 代码问题: {bottleneck_analysis.get('code_issues', [])}

请提供：
1. 针对{primary_bottleneck}瓶颈的具体优化方案
2. 代码修改的优先级排序
3. 预期的性能提升幅度
4. 实施的难度评估
5. 具体的代码修改建议

提供实用的优化建议：
"""
        
        return prompt
    
    def _parse_optimization_suggestions(self, llm_response: str, 
                                      bottleneck_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """解析优化建议"""
        
        # 基于瓶颈类型生成结构化建议
        primary_bottleneck = bottleneck_analysis.get("primary_bottleneck", "unknown")
        
        suggestions = {
            "optimization_strategy": self._get_optimization_strategy(primary_bottleneck),
            "priority_suggestions": self._extract_priority_suggestions(llm_response),
            "code_modifications": self._extract_code_modifications(llm_response),
            "expected_improvement": self._estimate_improvement(bottleneck_analysis),
            "implementation_difficulty": self._assess_difficulty(primary_bottleneck),
            "detailed_analysis": llm_response
        }
        
        return suggestions
    
    def _get_optimization_strategy(self, primary_bottleneck: str) -> str:
        """获取优化策略"""
        
        strategy_map = {
            "algorithm_inefficiency": "重新设计算法架构",
            "memory_bandwidth": "优化内存访问模式",
            "compute_utilization": "提高计算并行度",
            "minor_optimizations": "细节调优"
        }
        
        return strategy_map.get(primary_bottleneck, "综合优化")
    
    def _extract_priority_suggestions(self, llm_response: str) -> list:
        """提取优先级建议"""
        
        # 简化实现，实际可以使用更复杂的NLP解析
        suggestions = []
        
        if "内存" in llm_response or "memory" in llm_response.lower():
            suggestions.append("优化内存访问模式")
        if "并行" in llm_response or "parallel" in llm_response.lower():
            suggestions.append("增加并行度")
        if "BLOCK_SIZE" in llm_response:
            suggestions.append("调优BLOCK_SIZE参数")
        if "向量化" in llm_response or "vector" in llm_response.lower():
            suggestions.append("使用向量化操作")
        
        return suggestions if suggestions else ["根据LLM建议进行优化"]
    
    def _extract_code_modifications(self, llm_response: str) -> list:
        """提取代码修改建议"""
        
        modifications = []
        
        if "tl.load" in llm_response:
            modifications.append("优化内存加载操作")
        if "mask" in llm_response:
            modifications.append("改进边界条件处理")
        if "grid" in llm_response:
            modifications.append("调整kernel启动配置")
        
        return modifications if modifications else ["参考详细分析进行修改"]
    
    def _estimate_improvement(self, bottleneck_analysis: Dict[str, Any]) -> str:
        """估算改进幅度"""
        
        severity = bottleneck_analysis.get("bottleneck_severity", "medium")
        
        improvement_map = {
            "critical": "50-200%",
            "high": "20-80%", 
            "medium": "10-40%",
            "low": "5-20%"
        }
        
        return improvement_map.get(severity, "10-30%")
    
    def _assess_difficulty(self, primary_bottleneck: str) -> str:
        """评估实施难度"""
        
        difficulty_map = {
            "algorithm_inefficiency": "high",
            "memory_bandwidth": "medium",
            "compute_utilization": "medium",
            "minor_optimizations": "low"
        }
        
        return difficulty_map.get(primary_bottleneck, "medium")