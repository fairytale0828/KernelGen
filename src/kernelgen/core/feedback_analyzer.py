"""
错误分析和反馈生成器
分析编译错误、运行时错误和性能问题，生成优化建议
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeedbackResult:
    """反馈结果"""
    feedback_type: str  # "compilation_error", "runtime_error", "performance_issue", "success"
    error_message: str
    suggestions: List[str]
    optimization_hints: List[str]
    should_retry: bool
    confidence: float  # 0-1, 修复建议的置信度

class FeedbackAnalyzer:
    """错误分析和反馈生成器"""
    
    def __init__(self):
        self.compilation_patterns = self._init_compilation_patterns()
        self.runtime_patterns = self._init_runtime_patterns()
        self.performance_thresholds = {
            "min_speedup": 0.6,  # 最低加速比要求
            "target_speedup": 1.2,  # 目标加速比
            "excellent_speedup": 2.0  # 优秀加速比
        }
    
    def _init_compilation_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化编译错误模式"""
        return {
            "import_error": {
                "patterns": [
                    r"ModuleNotFoundError.*triton",
                    r"ImportError.*triton",
                    r"No module named.*triton"
                ],
                "suggestions": [
                    "确保正确导入triton模块: import triton, import triton.language as tl",
                    "检查triton是否正确安装",
                    "确保导入语句在文件开头"
                ]
            },
            "syntax_error": {
                "patterns": [
                    r"SyntaxError",
                    r"IndentationError",
                    r"invalid syntax"
                ],
                "suggestions": [
                    "检查Python语法错误，特别是缩进和括号匹配",
                    "确保所有函数定义和装饰器语法正确",
                    "检查是否有未闭合的括号或引号"
                ]
            },
            "triton_jit_error": {
                "patterns": [
                    r"@triton\.jit.*error",
                    r"triton\.jit.*not found",
                    r"__triton_jit__.*error"
                ],
                "suggestions": [
                    "确保@triton.jit装饰器语法正确",
                    "检查kernel函数参数类型注解",
                    "确保使用tl.constexpr标记编译时常量"
                ]
            },
            "type_annotation_error": {
                "patterns": [
                    r"tl\.constexpr.*error",
                    r"constexpr.*not defined",
                    r"BLOCK_SIZE.*constexpr"
                ],
                "suggestions": [
                    "使用tl.constexpr标记编译时常量: BLOCK_SIZE: tl.constexpr",
                    "确保所有模板参数都有正确的类型注解",
                    "检查triton.language导入是否正确"
                ]
            },
            "memory_access_error": {
                "patterns": [
                    r"load.*error",
                    r"store.*error",
                    r"pointer.*error",
                    r"offset.*error"
                ],
                "suggestions": [
                    "检查内存访问边界，确保使用mask防止越界",
                    "验证指针计算和偏移量是否正确",
                    "确保load和store操作的数据类型匹配"
                ]
            }
        }
    
    def _init_runtime_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化运行时错误模式"""
        return {
            "cuda_error": {
                "patterns": [
                    r"CUDA.*error",
                    r"RuntimeError.*CUDA",
                    r"device.*error"
                ],
                "suggestions": [
                    "检查CUDA设备是否可用和兼容",
                    "确保张量在正确的设备上",
                    "检查GPU内存是否足够"
                ]
            },
            "shape_mismatch": {
                "patterns": [
                    r"shape.*mismatch",
                    r"size.*mismatch",
                    r"dimension.*error"
                ],
                "suggestions": [
                    "检查输入和输出张量的形状是否匹配",
                    "验证kernel启动参数中的维度计算",
                    "确保reshape操作的正确性"
                ]
            },
            "kernel_launch_error": {
                "patterns": [
                    r"kernel.*launch.*error",
                    r"grid.*error",
                    r"block.*error"
                ],
                "suggestions": [
                    "检查kernel启动的grid和block配置",
                    "确保BLOCK_SIZE是2的幂次",
                    "验证kernel参数传递是否正确"
                ]
            },
            "numerical_error": {
                "patterns": [
                    r"nan.*detected",
                    r"inf.*detected",
                    r"numerical.*unstable"
                ],
                "suggestions": [
                    "检查数值计算的稳定性",
                    "添加数值范围检查和保护",
                    "考虑使用更稳定的数值算法"
                ]
            }
        }
    
    def analyze_compilation_error(self, error_message: str, kernel_code: str) -> FeedbackResult:
        """分析编译错误"""
        logger.debug(f"分析编译错误: {error_message[:200]}...")
        
        suggestions = []
        optimization_hints = []
        confidence = 0.0
        
        # 匹配错误模式
        for error_type, config in self.compilation_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, error_message, re.IGNORECASE):
                    suggestions.extend(config["suggestions"])
                    confidence = max(confidence, 0.8)
                    logger.debug(f"匹配到编译错误类型: {error_type}")
                    break
        
        # 代码分析
        code_suggestions = self._analyze_code_structure(kernel_code)
        suggestions.extend(code_suggestions)
        
        # 通用建议
        if not suggestions:
            suggestions = [
                "检查Triton语法是否正确",
                "确保所有必要的导入语句存在",
                "验证函数签名和参数类型",
                "检查装饰器和类型注解"
            ]
            confidence = 0.3
        
        return FeedbackResult(
            feedback_type="compilation_error",
            error_message=error_message,
            suggestions=suggestions,
            optimization_hints=optimization_hints,
            should_retry=True,
            confidence=confidence
        )
    
    def analyze_runtime_error(self, error_message: str, kernel_code: str, 
                            test_inputs: List[Any]) -> FeedbackResult:
        """分析运行时错误"""
        logger.debug(f"分析运行时错误: {error_message[:200]}...")
        
        suggestions = []
        optimization_hints = []
        confidence = 0.0
        
        # 匹配错误模式
        for error_type, config in self.runtime_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, error_message, re.IGNORECASE):
                    suggestions.extend(config["suggestions"])
                    confidence = max(confidence, 0.7)
                    logger.debug(f"匹配到运行时错误类型: {error_type}")
                    break
        
        # 输入分析
        input_suggestions = self._analyze_test_inputs(test_inputs, error_message)
        suggestions.extend(input_suggestions)
        
        # 通用建议
        if not suggestions:
            suggestions = [
                "检查kernel启动参数是否正确",
                "验证输入张量的设备和数据类型",
                "确保内存访问不会越界",
                "检查数值计算的稳定性"
            ]
            confidence = 0.4
        
        return FeedbackResult(
            feedback_type="runtime_error",
            error_message=error_message,
            suggestions=suggestions,
            optimization_hints=optimization_hints,
            should_retry=True,
            confidence=confidence
        )
    
    def analyze_correctness_error(self, pytorch_result: Any, triton_result: Any, 
                                tolerance: float = 1e-5) -> FeedbackResult:
        """分析正确性错误"""
        logger.debug("分析正确性错误...")
        
        try:
            import torch
            
            # 计算差异
            if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
                mean_diff = torch.mean(torch.abs(pytorch_result - triton_result)).item()
                
                suggestions = []
                optimization_hints = []
                
                if max_diff > tolerance:
                    if max_diff > 1.0:
                        suggestions.extend([
                            "结果差异很大，检查算法实现是否正确",
                            "验证数学运算的逻辑",
                            "检查是否有符号错误或运算顺序问题"
                        ])
                    elif max_diff > 0.01:
                        suggestions.extend([
                            "存在数值精度问题，检查浮点运算",
                            "考虑使用更高精度的数据类型",
                            "检查边界条件的处理"
                        ])
                    else:
                        suggestions.extend([
                            "轻微的数值差异，可能是浮点精度问题",
                            "检查运算顺序和舍入误差",
                            "考虑调整容差或使用更稳定的算法"
                        ])
                
                error_msg = f"最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}, 容差: {tolerance}"
                
                return FeedbackResult(
                    feedback_type="correctness_error",
                    error_message=error_msg,
                    suggestions=suggestions,
                    optimization_hints=optimization_hints,
                    should_retry=True,
                    confidence=0.6
                )
            
        except Exception as e:
            logger.error(f"正确性分析失败: {str(e)}")
        
        return FeedbackResult(
            feedback_type="correctness_error",
            error_message="无法比较结果",
            suggestions=["检查输出格式和数据类型是否匹配"],
            optimization_hints=[],
            should_retry=True,
            confidence=0.3
        )
    
    def analyze_performance(self, speedup: float, triton_time: float, 
                          pytorch_time: float, kernel_code: str) -> FeedbackResult:
        """分析性能问题"""
        logger.debug(f"分析性能: 加速比={speedup:.2f}x")
        
        suggestions = []
        optimization_hints = []
        should_retry = False
        confidence = 0.5
        
        if speedup >= self.performance_thresholds["excellent_speedup"]:
            # 性能优秀
            return FeedbackResult(
                feedback_type="success",
                error_message=f"优秀的性能: {speedup:.2f}x加速",
                suggestions=["性能已经很好，可以考虑进一步优化或结束"],
                optimization_hints=[],
                should_retry=False,
                confidence=0.9
            )
        
        elif speedup >= self.performance_thresholds["target_speedup"]:
            # 性能良好
            optimization_hints = self._generate_performance_optimization_hints(kernel_code, speedup)
            return FeedbackResult(
                feedback_type="success",
                error_message=f"良好的性能: {speedup:.2f}x加速",
                suggestions=["性能达标，可以尝试进一步优化"],
                optimization_hints=optimization_hints,
                should_retry=False,
                confidence=0.7
            )
        
        elif speedup >= self.performance_thresholds["min_speedup"]:
            # 性能一般，需要优化
            suggestions = [
                "性能略低于预期，建议优化",
                "检查内存访问模式是否高效",
                "考虑调整BLOCK_SIZE以提高GPU利用率"
            ]
            optimization_hints = self._generate_performance_optimization_hints(kernel_code, speedup)
            should_retry = True
            confidence = 0.6
        
        else:
            # 性能差，需要大幅优化
            suggestions = [
                "性能明显低于PyTorch，需要重新设计",
                "检查算法实现是否存在性能瓶颈",
                "考虑使用更高效的Triton编程模式",
                "检查是否有不必要的内存拷贝或同步"
            ]
            optimization_hints = self._generate_performance_optimization_hints(kernel_code, speedup)
            should_retry = True
            confidence = 0.8
        
        error_msg = f"加速比: {speedup:.2f}x (Triton: {triton_time:.3f}ms, PyTorch: {pytorch_time:.3f}ms)"
        
        return FeedbackResult(
            feedback_type="performance_issue",
            error_message=error_msg,
            suggestions=suggestions,
            optimization_hints=optimization_hints,
            should_retry=should_retry,
            confidence=confidence
        )
    
    def _analyze_code_structure(self, kernel_code: str) -> List[str]:
        """分析代码结构"""
        suggestions = []
        
        # 检查必要的导入
        if "import triton" not in kernel_code:
            suggestions.append("添加 'import triton' 导入")
        if "import triton.language as tl" not in kernel_code:
            suggestions.append("添加 'import triton.language as tl' 导入")
        if "import torch" not in kernel_code:
            suggestions.append("添加 'import torch' 导入")
        
        # 检查装饰器
        if "@triton.jit" not in kernel_code:
            suggestions.append("确保kernel函数有@triton.jit装饰器")
        
        # 检查常见模式
        if "tl.constexpr" not in kernel_code and "BLOCK_SIZE" in kernel_code:
            suggestions.append("使用tl.constexpr标记BLOCK_SIZE参数")
        
        if "tl.program_id" not in kernel_code:
            suggestions.append("使用tl.program_id获取程序ID")
        
        return suggestions
    
    def _analyze_test_inputs(self, test_inputs: List[Any], error_message: str) -> List[str]:
        """分析测试输入"""
        suggestions = []
        
        try:
            import torch
            
            for i, inp in enumerate(test_inputs):
                if isinstance(inp, torch.Tensor):
                    if not inp.is_cuda and "cuda" in error_message.lower():
                        suggestions.append(f"将输入张量{i}移动到CUDA设备")
                    
                    if inp.numel() == 0:
                        suggestions.append(f"输入张量{i}为空，检查形状")
                    
                    if torch.isnan(inp).any():
                        suggestions.append(f"输入张量{i}包含NaN值")
                    
                    if torch.isinf(inp).any():
                        suggestions.append(f"输入张量{i}包含无穷值")
        
        except Exception as e:
            logger.debug(f"输入分析失败: {str(e)}")
        
        return suggestions
    
    def _generate_performance_optimization_hints(self, kernel_code: str, current_speedup: float) -> List[str]:
        """生成性能优化建议"""
        hints = []
        
        # 基于当前性能水平的建议
        if current_speedup < 0.5:
            hints.extend([
                "考虑重新设计算法，当前实现可能存在根本性能问题",
                "检查是否有不必要的数据拷贝或类型转换",
                "验证内存访问模式是否合理"
            ])
        elif current_speedup < 1.0:
            hints.extend([
                "优化内存访问模式，使用连续内存访问",
                "调整BLOCK_SIZE以匹配GPU架构",
                "减少分支和条件语句"
            ])
        else:
            hints.extend([
                "尝试更大的BLOCK_SIZE以提高并行度",
                "考虑使用向量化操作",
                "优化循环展开和内存预取"
            ])
        
        # 基于代码分析的建议
        if "for " in kernel_code:
            hints.append("考虑将循环向量化或使用Triton的并行原语")
        
        if kernel_code.count("tl.load") > 3:
            hints.append("考虑合并多个内存加载操作")
        
        if "tl.store" in kernel_code and "mask" not in kernel_code:
            hints.append("使用mask优化边界条件处理")
        
        return hints
    
    def generate_optimization_prompt(self, feedback: FeedbackResult, 
                                   original_pytorch_code: str, 
                                   failed_kernel_code: str) -> str:
        """生成优化提示"""
        
        if feedback.feedback_type == "compilation_error":
            prompt_type = "编译错误修复"
            specific_instructions = f"""
编译失败，错误信息：
{feedback.error_message}

修复建议：
{chr(10).join(f"- {s}" for s in feedback.suggestions)}

请修复编译错误，确保代码能够成功编译。
"""
        
        elif feedback.feedback_type == "runtime_error":
            prompt_type = "运行时错误修复"
            specific_instructions = f"""
运行时失败，错误信息：
{feedback.error_message}

修复建议：
{chr(10).join(f"- {s}" for s in feedback.suggestions)}

请修复运行时错误，确保kernel能够正确执行。
"""
        
        elif feedback.feedback_type == "correctness_error":
            prompt_type = "正确性问题修复"
            specific_instructions = f"""
计算结果不正确，错误信息：
{feedback.error_message}

修复建议：
{chr(10).join(f"- {s}" for s in feedback.suggestions)}

请修复算法逻辑，确保计算结果与PyTorch一致。
"""
        
        elif feedback.feedback_type == "performance_issue":
            prompt_type = "性能优化"
            specific_instructions = f"""
性能需要优化，当前状态：
{feedback.error_message}

优化建议：
{chr(10).join(f"- {s}" for s in feedback.suggestions)}

性能优化提示：
{chr(10).join(f"- {h}" for h in feedback.optimization_hints)}

请优化kernel性能，提高相对于PyTorch的加速比。
"""
        
        else:
            return ""
        
        prompt = f"""你是一个Triton GPU kernel优化专家。需要{prompt_type}。

## 原始PyTorch实现：
```python
{original_pytorch_code}
```

## 当前有问题的Triton实现：
```python
{failed_kernel_code}
```

## 问题分析：
{specific_instructions}

## 要求：
1. 保持与PyTorch实现完全相同的功能
2. 修复所有错误，确保代码能正确运行
3. 优化性能，提高GPU利用率
4. 提供完整的、可直接运行的代码
5. 包含必要的wrapper函数和测试代码

请提供修复后的完整Triton kernel实现："""

        return prompt