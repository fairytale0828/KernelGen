"""
Kernel生成器核心类
集成数据集加载、LLM生成、性能测试等功能
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..dataset import KernelBenchLoader
from ..llm import LLMClient
from ..prompts import get_triton_generation_prompt, get_triton_optimization_prompt
from .performance_benchmark import TritonPerformanceBenchmark

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """生成结果数据类"""
    problem_id: int
    iteration: int
    success: bool
    kernel_code: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    generation_time: float = 0.0
    compilation_time: float = 0.0
    benchmark_time: float = 0.0

class KernelGenerator:
    """Kernel生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成器
        
        Args:
            config: 完整配置字典
        """
        self.config = config
        
        # 初始化组件
        self.dataset_loader = KernelBenchLoader(config["dataset"])
        self.llm_client = LLMClient(config["generation"]["llm"])
        self.benchmark = TritonPerformanceBenchmark(
            device=config["performance"]["device"],
            warmup_runs=config["performance"]["warmup_runs"],
            benchmark_runs=config["performance"]["benchmark_runs"]
        )
        
        # 生成配置
        self.max_iterations = config["generation"]["max_iterations"]
        self.early_stop_threshold = config["generation"]["early_stop_threshold"]
        self.min_successful_iterations = config["generation"]["min_successful_iterations"]
        self.backend = config["generation"]["backend"]
        
        # 输出配置
        self.output_config = config["output"]
        self.run_dir = os.path.join(
            self.output_config["base_dir"], 
            self.output_config["run_name"]
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 结果存储
        self.results = []
        self.best_results = {}  # problem_id -> best_result
        
        logger.info(f"Kernel生成器初始化完成")
        logger.info(f"输出目录: {self.run_dir}")
    
    def generate_single_kernel(self, problem_id: int, iteration: int) -> GenerationResult:
        """
        生成单个kernel
        
        Args:
            problem_id: 问题ID
            iteration: 迭代次数
            
        Returns:
            生成结果
        """
        result = GenerationResult(
            problem_id=problem_id,
            iteration=iteration,
            success=False
        )
        
        try:
            # 1. 获取问题信息
            problem_info = self.dataset_loader.get_problem_by_id(problem_id)
            pytorch_code = problem_info["code"]
            
            logger.info(f"开始生成 Problem {problem_id} Iteration {iteration}")
            
            # 2. 生成kernel代码
            start_time = time.time()
            
            if self.backend == "triton":
                prompt = get_triton_generation_prompt(pytorch_code)
                system_prompt = "You are an expert Triton GPU kernel programmer."
            else:
                raise ValueError(f"不支持的后端: {self.backend}")
            
            generated_text = self.llm_client.generate(prompt, system_prompt)
            kernel_code = self.llm_client.extract_code_block(generated_text, "python")
            
            if not kernel_code:
                result.error_message = "未能从生成文本中提取代码"
                return result
            
            result.kernel_code = kernel_code
            result.generation_time = time.time() - start_time
            
            # 3. 编译验证
            start_time = time.time()
            
            # 语法验证
            is_valid, error_msg = self.benchmark.validate_kernel_syntax(kernel_code)
            if not is_valid:
                result.error_message = f"语法验证失败: {error_msg}"
                return result
            
            # 编译测试
            compiled_func = self.benchmark.compile_and_load_kernel(
                kernel_code, f"kernel_{problem_id}_{iteration}"
            )
            
            if not compiled_func:
                result.error_message = "编译失败"
                return result
            
            result.compilation_time = time.time() - start_time
            
            # 4. 性能测试
            start_time = time.time()
            
            # 从PyTorch代码中提取测试参数
            test_shapes = self._extract_test_shapes(pytorch_code)
            
            if self._is_matmul_operation(pytorch_code):
                perf_results = self.benchmark.benchmark_matmul(compiled_func, test_shapes)
            else:
                perf_results = self.benchmark.benchmark_elementwise(compiled_func, test_shapes)
            
            result.benchmark_time = time.time() - start_time
            
            if not perf_results["success"]:
                result.error_message = f"性能测试失败: {perf_results.get('error', '未知错误')}"
                return result
            
            # 计算平均性能指标
            speedups = perf_results["speedups"]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                avg_triton_time = sum(perf_results["triton_times"]) / len(perf_results["triton_times"])
                avg_pytorch_time = sum(perf_results["pytorch_times"]) / len(perf_results["pytorch_times"])
                
                result.performance_metrics = {
                    "avg_speedup": avg_speedup,
                    "avg_triton_time_ms": avg_triton_time,
                    "avg_pytorch_time_ms": avg_pytorch_time,
                    "max_speedup": max(speedups),
                    "min_speedup": min(speedups)
                }
                
                # 添加GFLOPS信息（如果有）
                if "triton_gflops" in perf_results:
                    result.performance_metrics["avg_triton_gflops"] = sum(perf_results["triton_gflops"]) / len(perf_results["triton_gflops"])
                    result.performance_metrics["avg_pytorch_gflops"] = sum(perf_results["pytorch_gflops"]) / len(perf_results["pytorch_gflops"])
            
            result.success = True
            
            logger.info(f"Problem {problem_id} Iteration {iteration} 生成成功")
            if result.performance_metrics:
                logger.info(f"  平均加速比: {result.performance_metrics['avg_speedup']:.2f}x")
            
        except Exception as e:
            result.error_message = f"生成过程异常: {str(e)}"
            logger.error(f"Problem {problem_id} Iteration {iteration} 生成失败: {str(e)}")
        
        return result
    
    def generate_for_problem(self, problem_id: int) -> List[GenerationResult]:
        """
        为单个问题生成多个kernel
        
        Args:
            problem_id: 问题ID
            
        Returns:
            生成结果列表
        """
        problem_results = []
        successful_iterations = 0
        best_speedup = 0.0
        
        logger.info(f"开始为Problem {problem_id}生成kernel")
        
        for iteration in range(1, self.max_iterations + 1):
            result = self.generate_single_kernel(problem_id, iteration)
            problem_results.append(result)
            
            if result.success:
                successful_iterations += 1
                
                # 检查是否是最佳结果
                if result.performance_metrics:
                    speedup = result.performance_metrics["avg_speedup"]
                    if speedup > best_speedup:
                        best_speedup = speedup
                        self.best_results[problem_id] = result
                        logger.info(f"Problem {problem_id} 发现新的最佳kernel: {speedup:.2f}x")
                
                # 检查早停条件
                if (best_speedup >= self.early_stop_threshold and 
                    successful_iterations >= self.min_successful_iterations):
                    logger.info(f"Problem {problem_id} 达到早停条件: {best_speedup:.2f}x >= {self.early_stop_threshold:.2f}x")
                    break
            
            # 保存中间结果
            if iteration % 5 == 0:
                self._save_intermediate_results(problem_id, problem_results)
        
        logger.info(f"Problem {problem_id} 完成: {successful_iterations}/{len(problem_results)} 成功")
        return problem_results
    
    def generate_batch(self) -> Dict[str, Any]:
        """
        批量生成kernel
        
        Returns:
            批量生成结果摘要
        """
        # 加载数据集
        self.dataset_loader.load_dataset()
        target_problem_ids = self.dataset_loader.get_target_problem_ids()
        
        logger.info(f"开始批量生成，目标问题: {target_problem_ids}")
        
        # 保存配置
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        total_start_time = time.time()
        
        # 为每个问题生成kernel
        for problem_id in target_problem_ids:
            try:
                problem_results = self.generate_for_problem(problem_id)
                self.results.extend(problem_results)
                
                # 保存问题结果
                self._save_problem_results(problem_id, problem_results)
                
            except Exception as e:
                logger.error(f"Problem {problem_id} 生成失败: {str(e)}")
        
        total_time = time.time() - total_start_time
        
        # 生成摘要
        summary = self._generate_summary(total_time)
        
        # 保存最终结果
        self._save_final_results(summary)
        
        logger.info("批量生成完成")
        return summary
    
    def _extract_test_shapes(self, pytorch_code: str) -> List[Tuple]:
        """从PyTorch代码中提取测试形状"""
        # 简单的形状提取逻辑
        # 在实际实现中，可以更智能地解析代码
        
        # 默认形状
        default_shapes = [(1024, 1024, 1024)]
        
        try:
            # 查找常见的形状变量
            lines = pytorch_code.split('\n')
            shapes = []
            
            for line in lines:
                line = line.strip()
                if 'batch_size' in line and '=' in line:
                    # 尝试提取batch_size等参数
                    pass
            
            # 如果没有找到特定形状，返回默认值
            return default_shapes
            
        except Exception:
            return default_shapes
    
    def _is_matmul_operation(self, pytorch_code: str) -> bool:
        """判断是否是矩阵乘法操作"""
        matmul_indicators = [
            "torch.mm", "torch.matmul", "@", 
            "nn.Linear", "matrix_multiplication",
            "matmul", "gemm"
        ]
        
        code_lower = pytorch_code.lower()
        return any(indicator.lower() in code_lower for indicator in matmul_indicators)
    
    def _save_intermediate_results(self, problem_id: int, results: List[GenerationResult]):
        """保存中间结果"""
        if not self.output_config["save_logs"]:
            return
        
        results_data = [self._result_to_dict(r) for r in results]
        
        intermediate_path = os.path.join(
            self.run_dir, f"problem_{problem_id}_intermediate.json"
        )
        
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    def _save_problem_results(self, problem_id: int, results: List[GenerationResult]):
        """保存问题结果"""
        # 保存详细结果
        if self.output_config["save_logs"]:
            results_data = [self._result_to_dict(r) for r in results]
            
            results_path = os.path.join(
                self.run_dir, f"problem_{problem_id}_results.json"
            )
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 保存最佳kernel代码
        if problem_id in self.best_results:
            best_result = self.best_results[problem_id]
            
            if self.output_config["save_all_kernels"] or self.output_config["save_best_only"]:
                kernel_path = os.path.join(
                    self.run_dir, f"problem_{problem_id}_best_kernel.py"
                )
                
                with open(kernel_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Problem {problem_id} Best Kernel\n")
                    f.write(f"# Iteration: {best_result.iteration}\n")
                    f.write(f"# Speedup: {best_result.performance_metrics.get('avg_speedup', 0):.2f}x\n")
                    f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(best_result.kernel_code)
    
    def _save_final_results(self, summary: Dict[str, Any]):
        """保存最终结果"""
        # 保存摘要
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存所有结果
        if self.output_config["save_logs"]:
            all_results_path = os.path.join(self.run_dir, "all_results.json")
            all_results_data = [self._result_to_dict(r) for r in self.results]
            
            with open(all_results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results_data, f, indent=2, ensure_ascii=False)
    
    def _result_to_dict(self, result: GenerationResult) -> Dict[str, Any]:
        """将结果转换为字典"""
        return {
            "problem_id": result.problem_id,
            "iteration": result.iteration,
            "success": result.success,
            "kernel_code": result.kernel_code,
            "performance_metrics": result.performance_metrics,
            "error_message": result.error_message,
            "generation_time": result.generation_time,
            "compilation_time": result.compilation_time,
            "benchmark_time": result.benchmark_time
        }
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """生成结果摘要"""
        total_results = len(self.results)
        successful_results = len([r for r in self.results if r.success])
        
        # 按问题统计
        problem_stats = {}
        for result in self.results:
            pid = result.problem_id
            if pid not in problem_stats:
                problem_stats[pid] = {"total": 0, "successful": 0, "best_speedup": 0.0}
            
            problem_stats[pid]["total"] += 1
            if result.success:
                problem_stats[pid]["successful"] += 1
                if result.performance_metrics:
                    speedup = result.performance_metrics["avg_speedup"]
                    problem_stats[pid]["best_speedup"] = max(
                        problem_stats[pid]["best_speedup"], speedup
                    )
        
        # 计算平均性能
        successful_speedups = []
        for result in self.results:
            if result.success and result.performance_metrics:
                successful_speedups.append(result.performance_metrics["avg_speedup"])
        
        avg_speedup = sum(successful_speedups) / len(successful_speedups) if successful_speedups else 0.0
        max_speedup = max(successful_speedups) if successful_speedups else 0.0
        
        summary = {
            "config": self.config,
            "execution_time": total_time,
            "total_results": total_results,
            "successful_results": successful_results,
            "success_rate": successful_results / total_results if total_results > 0 else 0.0,
            "average_speedup": avg_speedup,
            "max_speedup": max_speedup,
            "problem_stats": problem_stats,
            "best_results_count": len(self.best_results),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary