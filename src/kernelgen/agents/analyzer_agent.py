"""
AnalyzerAgent - 分析和调试专家
基于aikg的Designer模式，负责分析PyTorch代码、设计Triton架构，以及根据错误信息进行调试
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalyzerAgent(BaseAgent):
    """
    分析和调试专家Agent (基于aikg的Designer模式)
    
    职责：
    1. 首次迭代：分析PyTorch代码，设计Triton kernel架构
    2. 后续迭代：根据错误信息和性能反馈进行调试分析
    3. 提供实现指导和优化建议
    4. 决定下一步的实现策略
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("AnalyzerAgent", llm_client, config)
        
        # 加载prompt模板
        self._load_prompt_templates()
        
        logger.info("AnalyzerAgent初始化完成")
    
    def _load_prompt_templates(self):
        """加载prompt模板"""
        # 首次分析模板
        self.initial_analysis_template = """你是一个专业的GPU kernel分析和架构设计专家。

## 任务：分析PyTorch操作并设计Triton Kernel架构

### PyTorch参考代码：
```python
{pytorch_code}
```

### 任务信息：
- 问题ID: {problem_id}
- 操作名称: {operation_name}
- 输入形状: {input_shapes}
- 输出形状: {output_shapes}

### 分析要求：

1. **操作分析**：
   - 识别核心计算模式（elementwise、reduction、matmul等）
   - 分析内存访问模式和数据依赖
   - 评估并行化潜力和计算复杂度

2. **架构设计**：
   - 推荐Triton kernel结构和组织方式
   - 设计内存布局和数据流策略
   - 选择合适的block size和grid配置

3. **实现指导**：
   - 定义关键实现步骤
   - 识别潜在瓶颈和优化机会
   - 提供边界条件处理建议

请按照以下JSON格式输出分析结果：

```json
{{
  "analysis_type": "initial_design",
  "operation_analysis": {{
    "operation_type": "操作类型",
    "computational_pattern": "计算模式描述",
    "memory_pattern": "内存访问模式",
    "parallelization_potential": "并行化潜力评估"
  }},
  "architecture_design": {{
    "kernel_structure": "kernel结构设计",
    "memory_layout": "内存布局策略", 
    "block_size_recommendation": "推荐的block size",
    "optimization_strategy": "优化策略"
  }},
  "implementation_guidance": {{
    "key_steps": ["步骤1", "步骤2", "步骤3"],
    "critical_considerations": ["考虑点1", "考虑点2"],
    "optimization_techniques": ["技术1", "技术2"]
  }}
}}
```"""

        # 调试分析模板
        self.debug_analysis_template = """你是一个专业的GPU kernel调试和优化专家。

## 任务：分析错误信息并提供调试指导

### 原始PyTorch代码：
```python
{pytorch_code}
```

### 当前生成的Triton代码：
```python
{generated_code}
```

### 错误信息：
{error_info}

### 性能结果（如有）：
{performance_info}

### 上次的设计方案：
{previous_design}

### 调试分析要求：

1. **错误诊断**：
   - 分析错误的根本原因
   - 识别代码中的问题点
   - 评估错误的严重程度

2. **修复策略**：
   - 提供具体的修复建议
   - 调整架构设计（如需要）
   - 优化实现策略

3. **性能优化**：
   - 分析性能瓶颈
   - 提供优化建议
   - 调整参数配置

请按照以下JSON格式输出调试结果：

```json
{{
  "analysis_type": "debug_analysis",
  "error_diagnosis": {{
    "error_type": "错误类型",
    "root_cause": "根本原因",
    "severity": "严重程度"
  }},
  "fix_strategy": {{
    "immediate_fixes": ["修复1", "修复2"],
    "architecture_adjustments": "架构调整建议",
    "implementation_changes": "实现修改建议"
  }},
  "updated_guidance": {{
    "key_steps": ["更新的步骤1", "更新的步骤2"],
    "critical_considerations": ["新的考虑点1", "新的考虑点2"],
    "optimization_techniques": ["优化技术1", "优化技术2"]
  }}
}}
```"""
    
    async def analyze_operation(self, 
                              pytorch_code: str,
                              problem_info: Dict[str, Any],
                              iteration: int = 1,
                              previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析PyTorch操作或调试现有实现
        
        Args:
            pytorch_code: PyTorch参考代码
            problem_info: 问题信息
            iteration: 当前迭代次数
            previous_results: 上次迭代的结果（包含错误信息、生成的代码等）
            
        Returns:
            分析结果字典
        """
        try:
            if iteration == 1:
                # 首次迭代：进行初始分析和设计
                return await self._initial_analysis(pytorch_code, problem_info)
            else:
                # 后续迭代：进行调试分析
                return await self._debug_analysis(pytorch_code, problem_info, previous_results)
                
        except Exception as e:
            logger.error(f"AnalyzerAgent执行失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _initial_analysis(self, pytorch_code: str, problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """首次分析和设计"""
        input_data = {
            "pytorch_code": pytorch_code,
            "problem_id": problem_info.get("problem_id", ""),
            "operation_name": problem_info.get("operation_name", ""),
            "input_shapes": str(problem_info.get("input_shapes", [])),
            "output_shapes": str(problem_info.get("output_shapes", []))
        }
        
        prompt = self.initial_analysis_template.format(**input_data)
        
        response = self.generate_llm_response(prompt)
        
        # 解析JSON响应
        result = self._extract_json_from_response(response)
        if result:
            logger.info("AnalyzerAgent初始分析完成")
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
    
    async def _debug_analysis(self, pytorch_code: str, problem_info: Dict[str, Any], 
                            previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """调试分析"""
        input_data = {
            "pytorch_code": pytorch_code,
            "generated_code": previous_results.get("generated_code", ""),
            "error_info": previous_results.get("error_info", ""),
            "performance_info": previous_results.get("performance_info", ""),
            "previous_design": str(previous_results.get("previous_design", {}))
        }
        
        prompt = self.debug_analysis_template.format(**input_data)
        
        response = self.generate_llm_response(prompt)
        
        # 解析JSON响应
        result = self._extract_json_from_response(response)
        if result:
            logger.info("AnalyzerAgent调试分析完成")
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
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Agent的主要执行方法，兼容BaseAgent的抽象方法
        
        Args:
            **kwargs: 关键字参数，包含pytorch_code, problem_info等
            
        Returns:
            分析结果字典
        """
        # 提取参数
        pytorch_code = kwargs.get("pytorch_code", "")
        problem_info = kwargs.get("problem_info", {})
        iteration = kwargs.get("iteration", 1)
        previous_results = kwargs.get("previous_results", None)
        
        # 调用异步方法（在同步上下文中）
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze_operation(pytorch_code, problem_info, iteration, previous_results)
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