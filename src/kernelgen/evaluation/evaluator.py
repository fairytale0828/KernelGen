"""
Kernel评估器
基于KernelBench的评估方法
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..dataset import KernelBenchLoader
from ..core.performance_benchmark import TritonPerformanceBenchmark

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    problem_id: int
    sample_id: int
    compiled: bool = False
    correctness: bool = False
    performance_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    runtime: float = -1.0
    
class KernelEvaluator:
    """Kernel评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化组件
        self.dataset_loader = KernelBenchLoader(config["dataset"])
        self.benchmark = TritonPerformanceBenchmark(
            device=config["performance"]["device"],
            warmup_runs=config["performance"]["warmup_runs"],
            benchmark_runs=config["performance"]["benchmark_runs"]
        )
        
        # 评估配置
        self.eval_config = config["evaluation"]
        self.num_correct_trials = self.eval_config["num_correct_trials"]
        self.num_perf_trials = self.eval_config["num_perf_trials"]
        self.pass_at_k_values = self.eval_config["pass_at_k_values"]
        
        # 输出配置
        self.output_config = config["output"]
        self.run_dir = os.path.join(
            self.output_config["base_dir"], 
            self.output_config["run_name"]
        )
        
        # 结果存储
        self.evaluation_results = []
        
        logger.info(f"Kernel评估器初始化完成")
    
    def evaluate_single_kernel(self, problem_id: int, kernel_code: str, 
                              sample_id: int = 0) -> EvaluationResult:
        """
        评估单个kernel
        
        Args:
            problem_id: 问题ID
            kernel_code: kernel代码
            sample_id: 样本ID
            
        Returns:
            评估结果
        """
        result = EvaluationResult(
            problem_id=problem_id,
            sample_id=sample_id
        )
        
        try:
            # 1. 获取参考代码
            problem_info = self.dataset_loader.get_problem_by_id(problem_id)
            pytorch_code = problem_info["code"]
            
            logger.debug(f"评估 Problem {problem_id} Sample {sample_id}")
            
            # 2. 编译测试
            start_time = time.time()
            
            # 语法验证
            is_valid, error_msg = self.benchmark.validate_kernel_syntax(kernel_code)
            if not is_valid:
                result.error_message = f"语法验证失败: {error_msg}"
                return result
            
            # 编译kernel
            compiled_func = self.benchmark.compile_and_load_kernel(
                kernel_code, f"eval_kernel_{problem_id}_{sample_id}"
            )
            
            if not compiled_func:
                result.error_message = "编译失败"
                return result
            
            result.compiled = True
            
            # 3. 正确性测试
            correctness_passed = self._test_correctness(
                compiled_func, pytorch_code, problem_id
            )
            result.correctness = correctness_passed
            
            if not correctness_passed:
                result.error_message = "正确性测试失败"
                # 即使正确性失败，也继续性能测试以获取更多信息
            
            # 4. 性能测试
            if result.compiled:
                perf_metrics = self._test_performance(
                    compiled_func, pytorch_code, problem_id
                )
                result.performance_metrics = perf_metrics
            
            result.runtime = time.time() - start_time
            
            logger.debug(f"Problem {problem_id} Sample {sample_id} 评估完成: "
                        f"编译={result.compiled}, 正确性={result.correctness}")
            
        except Exception as e:
            result.error_message = f"评估异常: {str(e)}"
            logger.error(f"Problem {problem_id} Sample {sample_id} 评估失败: {str(e)}")
        
        return result
    
    def evaluate_from_run_directory(self) -> Dict[str, Any]:
        """
        从运行目录评估已生成的kernel
        
        Returns:
            评估摘要
        """
        if not os.path.exists(self.run_dir):
            raise FileNotFoundError(f"运行目录不存在: {self.run_dir}")
        
        # 加载数据集
        self.dataset_loader.load_dataset()
        
        # 查找kernel文件
        kernel_files = []
        for filename in os.listdir(self.run_dir):
            if filename.endswith("_best_kernel.py"):
                kernel_files.append(filename)
        
        logger.info(f"找到 {len(kernel_files)} 个kernel文件进行评估")
        
        # 评估每个kernel
        for kernel_file in kernel_files:
            try:
                # 解析文件名获取问题ID
                problem_id = int(kernel_file.split("_")[1])
                
                # 读取kernel代码
                kernel_path = os.path.join(self.run_dir, kernel_file)
                with open(kernel_path, 'r', encoding='utf-8') as f:
                    kernel_code = f.read()
                
                # 评估kernel
                result = self.evaluate_single_kernel(problem_id, kernel_code)
                self.evaluation_results.append(result)
                
            except Exception as e:
                logger.error(f"评估文件 {kernel_file} 失败: {str(e)}")
        
        # 生成评估摘要
        summary = self._generate_evaluation_summary()
        
        # 保存评估结果
        self._save_evaluation_results(summary)
        
        return summary
    
    def evaluate_batch(self) -> Dict[str, Any]:
        """
        批量评估
        
        Returns:
            评估摘要
        """
        return self.evaluate_from_run_directory()
    
    def _test_correctness(self, compiled_func: Any, pytorch_code: str, 
                         problem_id: int) -> bool:
        """
        测试正确性
        
        Args:
            compiled_func: 编译后的函数
            pytorch_code: PyTorch参考代码
            problem_id: 问题ID
            
        Returns:
            是否通过正确性测试
        """
        try:
            # 创建PyTorch参考模型
            pytorch_model = self._create_pytorch_model(pytorch_code)
            if not pytorch_model:
                logger.warning(f"Problem {problem_id} 无法创建PyTorch参考模型")
                return False
            
            # 进行多次随机测试
            passed_trials = 0
            
            for trial in range(self.num_correct_trials):
                try:
                    # 生成随机输入
                    test_inputs = self._generate_test_inputs(pytorch_code)
                    
                    # PyTorch结果
                    with torch.no_grad():
                        pytorch_output = pytorch_model(*test_inputs)
                    
                    # Triton结果
                    triton_output = compiled_func(*test_inputs)
                    
                    # 比较结果
                    if self._compare_outputs(pytorch_output, triton_output):
                        passed_trials += 1
                    
                except Exception as e:
                    logger.debug(f"正确性测试trial {trial} 失败: {str(e)}")
                    continue
            
            # 需要通过大部分测试
            success_rate = passed_trials / self.num_correct_trials
            return success_rate >= 0.8  # 80%通过率
            
        except Exception as e:
            logger.error(f"正确性测试异常: {str(e)}")
            return False
    
    def _test_performance(self, compiled_func: Any, pytorch_code: str, 
                         problem_id: int) -> Optional[Dict[str, float]]:
        """
        测试性能
        
        Args:
            compiled_func: 编译后的函数
            pytorch_code: PyTorch参考代码
            problem_id: 问题ID
            
        Returns:
            性能指标字典
        """
        try:
            # 提取测试形状
            test_shapes = self._extract_test_shapes(pytorch_code)
            
            # 根据操作类型选择性能测试方法
            if self._is_matmul_operation(pytorch_code):
                perf_results = self.benchmark.benchmark_matmul(compiled_func, test_shapes)
            else:
                perf_results = self.benchmark.benchmark_elementwise(compiled_func, test_shapes)
            
            if not perf_results["success"]:
                logger.warning(f"Problem {problem_id} 性能测试失败: {perf_results.get('error')}")
                return None
            
            # 计算性能指标
            speedups = perf_results["speedups"]
            if speedups:
                metrics = {
                    "avg_speedup": sum(speedups) / len(speedups),
                    "max_speedup": max(speedups),
                    "min_speedup": min(speedups),
                    "avg_triton_time_ms": sum(perf_results["triton_times"]) / len(perf_results["triton_times"]),
                    "avg_pytorch_time_ms": sum(perf_results["pytorch_times"]) / len(perf_results["pytorch_times"])
                }
                
                # 添加GFLOPS信息（如果有）
                if "triton_gflops" in perf_results:
                    metrics["avg_triton_gflops"] = sum(perf_results["triton_gflops"]) / len(perf_results["triton_gflops"])
                    metrics["avg_pytorch_gflops"] = sum(perf_results["pytorch_gflops"]) / len(perf_results["pytorch_gflops"])
                
                return metrics
            
            return None
            
        except Exception as e:
            logger.error(f"性能测试异常: {str(e)}")
            return None
    
    def _create_pytorch_model(self, pytorch_code: str) -> Optional[Any]:
        """创建PyTorch参考模型"""
        try:
            # 执行PyTorch代码
            exec_globals = {}
            exec(pytorch_code, exec_globals)
            
            # 获取模型类和初始化参数
            model_class = exec_globals.get("Model")
            get_init_inputs = exec_globals.get("get_init_inputs")
            
            if not model_class or not get_init_inputs:
                return None
            
            # 创建模型实例
            init_inputs = get_init_inputs()
            model = model_class(*init_inputs)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.debug(f"创建PyTorch模型失败: {str(e)}")
            return None
    
    def _generate_test_inputs(self, pytorch_code: str) -> List[Any]:
        """生成测试输入"""
        try:
            # 执行PyTorch代码获取输入生成函数
            exec_globals = {}
            exec(pytorch_code, exec_globals)
            
            get_inputs = exec_globals.get("get_inputs")
            if not get_inputs:
                raise ValueError("未找到get_inputs函数")
            
            return get_inputs()
            
        except Exception as e:
            logger.debug(f"生成测试输入失败: {str(e)}")
            # 返回默认输入
            import torch
            return [torch.randn(1024, 512, device=self.benchmark.device)]
    
    def _compare_outputs(self, pytorch_output: Any, triton_output: Any, 
                        rtol: float = 1e-3, atol: float = 1e-3) -> bool:
        """比较输出结果"""
        try:
            import torch
            
            if isinstance(pytorch_output, (list, tuple)):
                if not isinstance(triton_output, (list, tuple)):
                    return False
                if len(pytorch_output) != len(triton_output):
                    return False
                
                for p_out, t_out in zip(pytorch_output, triton_output):
                    if not torch.allclose(p_out, t_out, rtol=rtol, atol=atol):
                        return False
                return True
            else:
                return torch.allclose(pytorch_output, triton_output, rtol=rtol, atol=atol)
                
        except Exception as e:
            logger.debug(f"输出比较失败: {str(e)}")
            return False
    
    def _extract_test_shapes(self, pytorch_code: str) -> List[Tuple]:
        """从PyTorch代码中提取测试形状"""
        # 简化实现，返回默认形状
        return [(1024, 1024, 1024)]
    
    def _is_matmul_operation(self, pytorch_code: str) -> bool:
        """判断是否是矩阵乘法操作"""
        matmul_indicators = [
            "torch.mm", "torch.matmul", "@", 
            "nn.Linear", "matrix_multiplication",
            "matmul", "gemm"
        ]
        
        code_lower = pytorch_code.lower()
        return any(indicator.lower() in code_lower for indicator in matmul_indicators)
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """生成评估摘要"""
        total_evaluated = len(self.evaluation_results)
        compiled_count = len([r for r in self.evaluation_results if r.compiled])
        correct_count = len([r for r in self.evaluation_results if r.correctness])
        
        # 计算pass@k
        pass_at_k_results = self._calculate_pass_at_k()
        
        # 性能统计
        performance_stats = self._calculate_performance_stats()
        
        summary = {
            "total_evaluated": total_evaluated,
            "compiled_count": compiled_count,
            "correct_count": correct_count,
            "compilation_success_rate": compiled_count / total_evaluated * 100 if total_evaluated > 0 else 0,
            "correctness_pass_rate": correct_count / total_evaluated * 100 if total_evaluated > 0 else 0,
            "pass_at_k": pass_at_k_results,
            "performance_stats": performance_stats,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def _calculate_pass_at_k(self) -> Dict[str, float]:
        """计算pass@k指标"""
        # 按问题分组
        problem_results = {}
        for result in self.evaluation_results:
            pid = result.problem_id
            if pid not in problem_results:
                problem_results[pid] = []
            problem_results[pid].append(result)
        
        pass_at_k = {}
        
        for k in self.pass_at_k_values:
            total_problems = len(problem_results)
            passed_problems = 0
            
            for pid, results in problem_results.items():
                # 计算该问题的通过数
                correct_results = [r for r in results if r.correctness and r.compiled]
                total_results = len(results)
                correct_count = len(correct_results)
                
                if total_results >= k:
                    # 计算pass@k概率
                    pass_prob = self._calc_pass_at_k(total_results, correct_count, k)
                    if pass_prob > 0.5:  # 简化判断
                        passed_problems += 1
            
            pass_at_k[f"pass@{k}"] = passed_problems / total_problems * 100 if total_problems > 0 else 0
        
        return pass_at_k
    
    def _calc_pass_at_k(self, n: int, c: int, k: int) -> float:
        """计算pass@k概率"""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    def _calculate_performance_stats(self) -> Dict[str, float]:
        """计算性能统计"""
        successful_results = [r for r in self.evaluation_results 
                            if r.performance_metrics is not None]
        
        if not successful_results:
            return {}
        
        speedups = [r.performance_metrics["avg_speedup"] for r in successful_results]
        
        return {
            "avg_speedup": sum(speedups) / len(speedups),
            "max_speedup": max(speedups),
            "min_speedup": min(speedups),
            "median_speedup": np.median(speedups),
            "std_speedup": np.std(speedups)
        }
    
    def _save_evaluation_results(self, summary: Dict[str, Any]):
        """保存评估结果"""
        # 保存详细结果
        results_path = os.path.join(self.run_dir, "evaluation_results.json")
        detailed_results = [
            {
                "problem_id": r.problem_id,
                "sample_id": r.sample_id,
                "compiled": r.compiled,
                "correctness": r.correctness,
                "performance_metrics": r.performance_metrics,
                "error_message": r.error_message,
                "runtime": r.runtime
            }
            for r in self.evaluation_results
        ]
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 保存摘要
        summary_path = os.path.join(self.run_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {self.run_dir}")