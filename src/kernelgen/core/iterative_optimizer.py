"""
迭代优化控制器
实现基于反馈的迭代优化循环
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .feedback_analyzer import FeedbackAnalyzer, FeedbackResult
from ..llm import LLMClient
from ..prompts import get_triton_generation_prompt, get_triton_optimization_prompt
from .performance_benchmark import TritonPerformanceBenchmark

logger = logging.getLogger(__name__)

@dataclass
class IterationResult:
    """单次迭代结果"""
    iteration: int
    kernel_code: str
    compilation_success: bool
    runtime_success: bool
    correctness_success: bool
    performance_metrics: Optional[Dict[str, float]]
    feedback: Optional[FeedbackResult]
    error_message: str
    generation_time: float
    validation_time: float

class IterativeOptimizer:
    """迭代优化控制器"""
    
    def __init__(self, llm_client: LLMClient, benchmark: TritonPerformanceBenchmark):
        """
        初始化迭代优化器
        
        Args:
            llm_client: LLM客户端
            benchmark: 性能基准测试器
        """
        self.llm_client = llm_client
        self.benchmark = benchmark
        self.feedback_analyzer = FeedbackAnalyzer()
        
        # 迭代控制参数
        self.max_iterations = 10
        self.early_stop_threshold = 1.2
        self.min_successful_iterations = 2
        self.max_consecutive_failures = 3
        
        # 结果存储
        self.iteration_history: List[IterationResult] = []
        self.best_result: Optional[IterationResult] = None
        self.best_speedup = 0.0
        
        logger.info("迭代优化器初始化完成")
    
    def optimize_kernel(self, pytorch_code: str, problem_id: int, 
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        迭代优化kernel
        
        Args:
            pytorch_code: PyTorch参考代码
            problem_id: 问题ID
            config: 配置参数
            
        Returns:
            优化结果摘要
        """
        logger.info(f"开始迭代优化 Problem {problem_id}")
        
        # 更新配置
        self.max_iterations = config.get("max_iterations", 10)
        self.early_stop_threshold = config.get("early_stop_threshold", 1.2)
        self.min_successful_iterations = config.get("min_successful_iterations", 2)
        
        # 重置状态
        self.iteration_history = []
        self.best_result = None
        self.best_speedup = 0.0
        
        # 创建PyTorch参考函数和测试输入
        pytorch_func, test_inputs = self._prepare_reference(pytorch_code)
        if not pytorch_func or not test_inputs:
            return self._create_failure_summary("无法创建PyTorch参考函数或测试输入")
        
        consecutive_failures = 0
        successful_iterations = 0
        
        # 第一轮：初始生成
        logger.info("🚀 第1轮：初始生成")
        result = self._initial_generation(pytorch_code, pytorch_func, test_inputs, 1)
        self.iteration_history.append(result)
        
        if result.correctness_success:
            successful_iterations += 1
            consecutive_failures = 0
            self._update_best_result(result)
            
            # 检查早停条件
            if self._should_early_stop(result, successful_iterations):
                logger.info(f"🎯 达到早停条件，加速比: {result.performance_metrics.get('speedup', 0):.2f}x")
                return self._create_success_summary()
        else:
            consecutive_failures += 1
        
        # 后续轮次：基于反馈优化
        for iteration in range(2, self.max_iterations + 1):
            logger.info(f"🔄 第{iteration}轮：基于反馈优化")
            
            # 检查是否应该停止
            if consecutive_failures >= self.max_consecutive_failures:
                logger.warning(f"连续{consecutive_failures}次失败，停止迭代")
                break
            
            # 基于反馈生成优化版本
            result = self._feedback_optimization(
                pytorch_code, pytorch_func, test_inputs, iteration
            )
            self.iteration_history.append(result)
            
            if result.correctness_success:
                successful_iterations += 1
                consecutive_failures = 0
                self._update_best_result(result)
                
                # 检查早停条件
                if self._should_early_stop(result, successful_iterations):
                    logger.info(f"🎯 达到早停条件，加速比: {result.performance_metrics.get('speedup', 0):.2f}x")
                    break
            else:
                consecutive_failures += 1
            
            # 显示进度
            self._log_progress(iteration, successful_iterations)
        
        return self._create_success_summary()
    
    def _initial_generation(self, pytorch_code: str, pytorch_func: Any, 
                          test_inputs: List[Any], iteration: int) -> IterationResult:
        """初始生成"""
        start_time = time.time()
        
        # 生成初始kernel
        prompt = get_triton_generation_prompt(pytorch_code)
        system_prompt = "You are an expert Triton GPU kernel programmer. Generate high-performance, correct Triton kernels."
        
        try:
            generated_text = self.llm_client.generate(prompt, system_prompt)
            kernel_code = self.llm_client.extract_code_block(generated_text, "python")
            
            if not kernel_code:
                return IterationResult(
                    iteration=iteration,
                    kernel_code="",
                    compilation_success=False,
                    runtime_success=False,
                    correctness_success=False,
                    performance_metrics=None,
                    feedback=None,
                    error_message="未能从生成文本中提取代码",
                    generation_time=time.time() - start_time,
                    validation_time=0.0
                )
            
            generation_time = time.time() - start_time
            
            # 验证和测试
            validation_start = time.time()
            result = self._validate_kernel(kernel_code, pytorch_func, test_inputs, iteration)
            result.generation_time = generation_time
            result.validation_time = time.time() - validation_start
            
            return result
            
        except Exception as e:
            return IterationResult(
                iteration=iteration,
                kernel_code="",
                compilation_success=False,
                runtime_success=False,
                correctness_success=False,
                performance_metrics=None,
                feedback=None,
                error_message=f"生成失败: {str(e)}",
                generation_time=time.time() - start_time,
                validation_time=0.0
            )
    
    def _feedback_optimization(self, pytorch_code: str, pytorch_func: Any, 
                             test_inputs: List[Any], iteration: int) -> IterationResult:
        """基于反馈的优化"""
        start_time = time.time()
        
        # 获取最佳反馈用于优化
        feedback = self._get_best_feedback_for_optimization()
        if not feedback:
            # 如果没有有用的反馈，使用通用优化提示
            feedback = self._create_generic_optimization_feedback()
        
        # 获取最佳kernel作为基础
        base_kernel = self.best_result.kernel_code if self.best_result else self._get_last_valid_kernel()
        
        # 生成优化提示
        optimization_prompt = self.feedback_analyzer.generate_optimization_prompt(
            feedback, pytorch_code, base_kernel
        )
        
        try:
            generated_text = self.llm_client.generate(optimization_prompt)
            kernel_code = self.llm_client.extract_code_block(generated_text, "python")
            
            if not kernel_code:
                return IterationResult(
                    iteration=iteration,
                    kernel_code="",
                    compilation_success=False,
                    runtime_success=False,
                    correctness_success=False,
                    performance_metrics=None,
                    feedback=feedback,
                    error_message="未能从优化文本中提取代码",
                    generation_time=time.time() - start_time,
                    validation_time=0.0
                )
            
            generation_time = time.time() - start_time
            
            # 验证和测试
            validation_start = time.time()
            result = self._validate_kernel(kernel_code, pytorch_func, test_inputs, iteration)
            result.generation_time = generation_time
            result.validation_time = time.time() - validation_start
            result.feedback = feedback
            
            return result
            
        except Exception as e:
            return IterationResult(
                iteration=iteration,
                kernel_code="",
                compilation_success=False,
                runtime_success=False,
                correctness_success=False,
                performance_metrics=None,
                feedback=feedback,
                error_message=f"优化生成失败: {str(e)}",
                generation_time=time.time() - start_time,
                validation_time=0.0
            )
    
    def _validate_kernel(self, kernel_code: str, pytorch_func: Any, 
                        test_inputs: List[Any], iteration: int) -> IterationResult:
        """验证kernel"""
        result = IterationResult(
            iteration=iteration,
            kernel_code=kernel_code,
            compilation_success=False,
            runtime_success=False,
            correctness_success=False,
            performance_metrics=None,
            feedback=None,
            error_message="",
            generation_time=0.0,
            validation_time=0.0
        )
        
        try:
            # 1. 语法验证
            is_valid, error_msg = self.benchmark.validate_kernel_syntax(kernel_code)
            if not is_valid:
                result.error_message = f"语法验证失败: {error_msg}"
                result.feedback = self.feedback_analyzer.analyze_compilation_error(error_msg, kernel_code)
                return result
            
            # 2. 编译测试
            compiled_func = self.benchmark.compile_and_load_kernel(kernel_code, f"kernel_{iteration}")
            if not compiled_func:
                result.error_message = "编译失败"
                result.feedback = self.feedback_analyzer.analyze_compilation_error("编译失败", kernel_code)
                return result
            
            result.compilation_success = True
            logger.debug(f"第{iteration}轮编译成功")
            
            # 3. 运行时测试
            try:
                triton_result = compiled_func(*test_inputs)
                result.runtime_success = True
                logger.debug(f"第{iteration}轮运行成功")
            except Exception as e:
                result.error_message = f"运行时错误: {str(e)}"
                result.feedback = self.feedback_analyzer.analyze_runtime_error(str(e), kernel_code, test_inputs)
                return result
            
            # 4. 正确性验证
            try:
                import torch
                pytorch_result = pytorch_func(*test_inputs)
                
                # 比较结果
                if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                    is_correct = torch.allclose(pytorch_result, triton_result, rtol=1e-4, atol=1e-5)
                    if is_correct:
                        result.correctness_success = True
                        logger.debug(f"第{iteration}轮正确性验证通过")
                    else:
                        result.error_message = "计算结果不正确"
                        result.feedback = self.feedback_analyzer.analyze_correctness_error(
                            pytorch_result, triton_result
                        )
                        return result
                else:
                    result.error_message = "结果类型不匹配"
                    result.feedback = self.feedback_analyzer.analyze_correctness_error(
                        pytorch_result, triton_result
                    )
                    return result
            
            except Exception as e:
                result.error_message = f"正确性验证失败: {str(e)}"
                result.feedback = self.feedback_analyzer.analyze_runtime_error(str(e), kernel_code, test_inputs)
                return result
            
            # 5. 性能测试
            try:
                perf_results = self.benchmark.benchmark_general(compiled_func, pytorch_func, test_inputs)
                
                if perf_results["success"] and perf_results["speedups"]:
                    speedup = perf_results["speedups"][0]
                    triton_time = perf_results["triton_times"][0]
                    pytorch_time = perf_results["pytorch_times"][0]
                    
                    result.performance_metrics = {
                        "speedup": speedup,
                        "triton_time_ms": triton_time,
                        "pytorch_time_ms": pytorch_time
                    }
                    
                    # 性能分析
                    result.feedback = self.feedback_analyzer.analyze_performance(
                        speedup, triton_time, pytorch_time, kernel_code
                    )
                    
                    logger.info(f"第{iteration}轮性能测试: {speedup:.2f}x加速")
                else:
                    result.error_message = f"性能测试失败: {perf_results.get('error', '未知错误')}"
                    
            except Exception as e:
                result.error_message = f"性能测试异常: {str(e)}"
                logger.warning(f"性能测试失败: {str(e)}")
        
        except Exception as e:
            result.error_message = f"验证过程异常: {str(e)}"
            logger.error(f"验证过程异常: {str(e)}")
        
        return result
    
    def _prepare_reference(self, pytorch_code: str) -> Tuple[Optional[Any], Optional[List[Any]]]:
        """准备PyTorch参考函数和测试输入"""
        try:
            # 执行PyTorch代码
            exec_globals = {}
            exec(pytorch_code, exec_globals)
            
            # 获取模型类和函数
            model_class = exec_globals.get("Model")
            get_init_inputs = exec_globals.get("get_init_inputs")
            get_inputs = exec_globals.get("get_inputs")
            
            if not all([model_class, get_init_inputs, get_inputs]):
                logger.error("PyTorch代码缺少必要的组件")
                return None, None
            
            # 创建模型实例
            init_inputs = get_init_inputs()
            model = model_class(*init_inputs)
            model.eval()
            
            # 生成测试输入
            test_inputs = get_inputs()
            
            # 将输入移动到GPU
            if self.benchmark.device == "cuda":
                import torch
                test_inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in test_inputs]
            
            return model, test_inputs
            
        except Exception as e:
            logger.error(f"准备参考函数失败: {str(e)}")
            return None, None
    
    def _update_best_result(self, result: IterationResult):
        """更新最佳结果"""
        if not result.correctness_success:
            return
        
        current_speedup = 0.0
        if result.performance_metrics:
            current_speedup = result.performance_metrics.get("speedup", 0.0)
        
        if current_speedup > self.best_speedup:
            self.best_speedup = current_speedup
            self.best_result = result
            logger.info(f"🏆 发现更好的kernel: {current_speedup:.2f}x加速")
    
    def _should_early_stop(self, result: IterationResult, successful_iterations: int) -> bool:
        """检查是否应该早停"""
        if not result.performance_metrics:
            return False
        
        speedup = result.performance_metrics.get("speedup", 0.0)
        return (speedup >= self.early_stop_threshold and 
                successful_iterations >= self.min_successful_iterations)
    
    def _get_best_feedback_for_optimization(self) -> Optional[FeedbackResult]:
        """获取最佳反馈用于优化"""
        # 优先使用最近的失败反馈
        for result in reversed(self.iteration_history):
            if result.feedback and result.feedback.should_retry:
                return result.feedback
        
        # 如果没有失败反馈，使用最佳结果的性能反馈
        if self.best_result and self.best_result.feedback:
            return self.best_result.feedback
        
        return None
    
    def _create_generic_optimization_feedback(self) -> FeedbackResult:
        """创建通用优化反馈"""
        return FeedbackResult(
            feedback_type="performance_issue",
            error_message="通用性能优化",
            suggestions=[
                "优化内存访问模式",
                "调整BLOCK_SIZE以提高并行度",
                "减少不必要的计算和内存操作"
            ],
            optimization_hints=[
                "使用更高效的Triton原语",
                "考虑向量化操作",
                "优化循环结构"
            ],
            should_retry=True,
            confidence=0.5
        )
    
    def _get_last_valid_kernel(self) -> str:
        """获取最后一个有效的kernel"""
        for result in reversed(self.iteration_history):
            if result.compilation_success and result.kernel_code:
                return result.kernel_code
        
        # 如果没有有效的kernel，返回空字符串
        return ""
    
    def _log_progress(self, iteration: int, successful_iterations: int):
        """记录进度"""
        total_iterations = len(self.iteration_history)
        success_rate = successful_iterations / total_iterations * 100
        
        logger.info(f"📊 进度更新:")
        logger.info(f"   已完成: {iteration}/{self.max_iterations}")
        logger.info(f"   成功率: {successful_iterations}/{total_iterations} ({success_rate:.1f}%)")
        logger.info(f"   最佳加速比: {self.best_speedup:.2f}x")
    
    def _create_success_summary(self) -> Dict[str, Any]:
        """创建成功摘要"""
        total_iterations = len(self.iteration_history)
        successful_iterations = len([r for r in self.iteration_history if r.correctness_success])
        
        return {
            "success": True,
            "total_iterations": total_iterations,
            "successful_iterations": successful_iterations,
            "success_rate": successful_iterations / total_iterations * 100 if total_iterations > 0 else 0,
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_result.kernel_code if self.best_result else None,
            "best_performance": self.best_result.performance_metrics if self.best_result else None,
            "iteration_history": [
                {
                    "iteration": r.iteration,
                    "compilation_success": r.compilation_success,
                    "runtime_success": r.runtime_success,
                    "correctness_success": r.correctness_success,
                    "performance_metrics": r.performance_metrics,
                    "error_message": r.error_message,
                    "generation_time": r.generation_time,
                    "validation_time": r.validation_time
                }
                for r in self.iteration_history
            ]
        }
    
    def _create_failure_summary(self, error_message: str) -> Dict[str, Any]:
        """创建失败摘要"""
        return {
            "success": False,
            "error_message": error_message,
            "total_iterations": 0,
            "successful_iterations": 0,
            "success_rate": 0.0,
            "best_speedup": 0.0,
            "best_kernel": None,
            "iteration_history": []
        }