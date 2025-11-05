"""
Design Agent - 架构设计师
"""

import logging
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DesignAgent(BaseAgent):
    """
    设计Agent - 负责分析算子需求并设计实现架构
    
    职责：
    1. 分析PyTorch算子的计算逻辑和数据流
    2. 设计Triton kernel的整体架构
    3. 根据优化建议调整设计方案
    4. 根据调试反馈修复设计缺陷
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("DesignAgent", llm_client, config)
        
        # 算子分类规则
        self.operator_patterns = {
            "elementwise": ["relu", "sigmoid", "tanh", "gelu", "add", "mul", "div"],
            "reduction": ["sum", "mean", "max", "min", "norm", "softmax"],
            "matmul": ["matmul", "linear", "bmm", "mm"],
            "conv": ["conv1d", "conv2d", "conv3d"],
            "attention": ["attention", "transformer", "mha"]
        }
    
    async def run(self, task_info: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        执行设计生成
        
        Args:
            task_info: 任务信息字典，包含当前所有代码和状态
            
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        try:
            # 从task_info中获取信息
            pytorch_code = task_info.get("pytorch_code", "")
            problem_info = task_info.get("problem_info", {})
            optimization_feedback = task_info.get("optimization_feedback", "")
            debug_feedback = task_info.get("debug_feedback", "")
            
            # 分析算子类型
            operator_analysis = self._analyze_operator(pytorch_code, problem_info)
            
            # 生成设计方案
            design_plan = await self._generate_design_plan(
                pytorch_code, operator_analysis, optimization_feedback, debug_feedback
            )
            
            # 生成实现指导
            implementation_guide = await self._generate_implementation_guide(
                design_plan, operator_analysis
            )
            
            # 构造返回结果
            result_content = {
                "design_plan": design_plan,
                "implementation_guide": implementation_guide,
                "operator_analysis": operator_analysis
            }
            
            # 格式化为JSON字符串
            import json
            formatted_result = json.dumps(result_content, indent=2, ensure_ascii=False)
            
            return formatted_result, "", ""
            
        except Exception as e:
            logger.error(f"设计生成失败: {str(e)}")
            raise
    
    def _analyze_operator(self, pytorch_code: str, problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析算子特征"""
        
        # 基于代码内容和问题信息分析算子类型
        operator_type = "unknown"
        for op_type, patterns in self.operator_patterns.items():
            for pattern in patterns:
                if pattern.lower() in pytorch_code.lower() or pattern.lower() in problem_info.get("name", "").lower():
                    operator_type = op_type
                    break
            if operator_type != "unknown":
                break
        
        # 分析计算复杂度
        complexity = self._estimate_computational_complexity(pytorch_code)
        
        # 分析内存访问模式
        memory_pattern = self._analyze_memory_pattern(pytorch_code)
        
        # 分析并行化潜力
        parallelization = self._analyze_parallelization_potential(pytorch_code, operator_type)
        
        return {
            "operator_type": operator_type,
            "computational_complexity": complexity,
            "memory_access_pattern": memory_pattern,
            "parallelization_potential": parallelization,
            "input_shapes": self._extract_input_shapes(pytorch_code),
            "output_shapes": self._extract_output_shapes(pytorch_code),
            "data_types": self._extract_data_types(pytorch_code)
        }
    
    async def _generate_design_plan(self, pytorch_code: str, operator_analysis: Dict[str, Any],
                                  optimization_feedback: str, debug_feedback: str) -> Dict[str, Any]:
        """生成设计方案"""
        
        # 构造设计提示
        design_prompt = self._create_design_prompt(
            pytorch_code, operator_analysis, optimization_feedback, debug_feedback
        )
        
        # 调用LLM生成设计
        design_response = self.generate_llm_response(
            design_prompt,
            "You are an expert GPU kernel architect. Analyze the PyTorch code and create a detailed Triton kernel design plan."
        )
        
        # 解析设计响应
        design_plan = self._parse_design_response(design_response, operator_analysis)
        
        return design_plan
    
    async def _generate_implementation_guide(self, design_plan: Dict[str, Any], 
                                           operator_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成实现指导"""
        
        guide_prompt = self._create_implementation_guide_prompt(design_plan, operator_analysis)
        
        guide_response = self.generate_llm_response(
            guide_prompt,
            "You are an expert Triton programmer. Provide detailed implementation guidance for the kernel design."
        )
        
        implementation_guide = self._parse_implementation_guide(guide_response, design_plan)
        
        return implementation_guide
    
    def _create_design_prompt(self, pytorch_code: str, operator_analysis: Dict[str, Any],
                            optimization_feedback: str, debug_feedback: str) -> str:
        """创建设计提示"""
        
        feedback_section = ""
        if debug_feedback:
            feedback_section += f"\n## 调试反馈：\n{debug_feedback}"
        if optimization_feedback:
            feedback_section += f"\n## 优化建议：\n{optimization_feedback}"
        
        prompt = f"""
分析以下PyTorch算子并设计高效的Triton kernel架构：

## PyTorch代码：
```python
{pytorch_code}
```

## 算子分析：
- 类型: {operator_analysis['operator_type']}
- 计算复杂度: {operator_analysis['computational_complexity']}
- 内存访问模式: {operator_analysis['memory_access_pattern']}
- 并行化潜力: {operator_analysis['parallelization_potential']}

{feedback_section}

## 设计要求：
1. 分析算子的核心计算逻辑
2. 设计合适的内存访问策略
3. 确定最优的并行化方案
4. 选择合适的BLOCK_SIZE
5. 考虑边界条件处理
6. 设计kernel启动参数

请提供详细的设计方案，包括：
- 整体架构设计
- 内存布局策略
- 并行化方案
- 性能优化考虑
- 实现难点分析
"""
        
        return prompt
    
    def _create_implementation_guide_prompt(self, design_plan: Dict[str, Any], 
                                          operator_analysis: Dict[str, Any]) -> str:
        """创建实现指导提示"""
        
        prompt = f"""
基于以下设计方案，提供详细的Triton kernel实现指导：

## 设计方案：
{design_plan}

## 算子分析：
{operator_analysis}

## 实现指导要求：
1. 详细的代码结构建议
2. 关键函数和变量命名
3. 内存访问的具体实现
4. 边界条件的处理方法
5. 性能关键点的实现技巧
6. 常见错误的避免方法

请提供具体的实现指导，包括：
- 代码框架结构
- 关键实现步骤
- 性能优化技巧
- 调试建议
- 测试策略
"""
        
        return prompt
    
    def _parse_design_response(self, response: str, operator_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """解析设计响应"""
        
        # 简化实现，实际可以使用更复杂的解析逻辑
        return {
            "architecture": "block_parallel",
            "memory_strategy": "coalesced_access",
            "parallelization": "thread_block",
            "recommended_block_size": 256,
            "memory_pattern": operator_analysis["memory_access_pattern"],
            "optimization_focus": ["memory_bandwidth", "compute_utilization"],
            "implementation_complexity": "medium",
            "design_details": response
        }
    
    def _parse_implementation_guide(self, response: str, design_plan: Dict[str, Any]) -> Dict[str, Any]:
        """解析实现指导"""
        
        return {
            "code_structure": "kernel_with_wrapper",
            "key_functions": ["main_kernel", "launch_wrapper"],
            "memory_management": "automatic",
            "boundary_handling": "mask_based",
            "performance_tips": ["use_vectorization", "optimize_memory_access"],
            "debugging_hints": ["check_shapes", "validate_outputs"],
            "implementation_details": response
        }
    
    # 辅助分析方法
    def _estimate_computational_complexity(self, code: str) -> str:
        """估算计算复杂度"""
        if any(op in code.lower() for op in ["matmul", "mm", "bmm"]):
            return "O(n^3)"
        elif any(op in code.lower() for op in ["conv", "convolution"]):
            return "O(n^2)"
        elif any(op in code.lower() for op in ["sum", "mean", "max", "min"]):
            return "O(n)"
        else:
            return "O(1)"
    
    def _analyze_memory_pattern(self, code: str) -> str:
        """分析内存访问模式"""
        if "matmul" in code.lower():
            return "matrix_access"
        elif any(op in code.lower() for op in ["sum", "mean"]):
            return "reduction"
        else:
            return "elementwise"
    
    def _analyze_parallelization_potential(self, code: str, operator_type: str) -> str:
        """分析并行化潜力"""
        if operator_type == "elementwise":
            return "high"
        elif operator_type == "matmul":
            return "medium"
        elif operator_type == "reduction":
            return "low"
        else:
            return "medium"
    
    def _extract_input_shapes(self, code: str) -> list:
        """提取输入形状信息"""
        return ["dynamic"]
    
    def _extract_output_shapes(self, code: str) -> list:
        """提取输出形状信息"""
        return ["dynamic"]
    
    def _extract_data_types(self, code: str) -> list:
        """提取数据类型信息"""
        if "float32" in code:
            return ["float32"]
        elif "float16" in code:
            return ["float16"]
        else:
            return ["float32"]