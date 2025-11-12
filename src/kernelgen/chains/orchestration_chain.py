"""
编排链 - 协调多个链的执行
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from .analysis_chain import AnalysisChain
from .generation_chain import GenerationChain
from .validation_chain import ValidationChain
from ..tools.performance_tools import PerformanceBenchmarkTool
from ..database import KernelBenchLoader
from ..core.performance_benchmark import TritonPerformanceBenchmark

logger = logging.getLogger(__name__)

@dataclass
class IterationResult:
    """单次迭代结果"""
    iteration: int
    analysis_result: Optional[Dict[str, Any]] = None
    generation_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    final_code: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    success: bool = False
    error_message: str = ""

class OrchestrationChain:
    """编排链 - 协调分析、生成、验证链的执行"""
    
    def __init__(self, llm: BaseChatModel, config: Dict[str, Any]):
        """
        初始化编排链
        
        Args:
            llm: LangChain聊天模型
            config: 配置信息
        """
        self.llm = llm
        self.config = config
        
        # 初始化各个链
        self.analysis_chain = AnalysisChain(llm)
        self.generation_chain = GenerationChain(llm)
        self.validation_chain = ValidationChain(llm)
        
        # 初始化性能测试工具
        performance_config = config.get("performance", {})
        self.benchmark = TritonPerformanceBenchmark(
            device=performance_config.get("device", "cuda"),
            warmup_runs=performance_config.get("warmup_runs", 10),
            benchmark_runs=performance_config.get("benchmark_runs", 100)
        )
        self.performance_tool = PerformanceBenchmarkTool(self.benchmark)
        
        # 迭代控制参数
        self.max_iterations = config.get("generation", {}).get("max_iterations", 5)
        self.early_stop_threshold = config.get("generation", {}).get("early_stop_threshold", 1.2)
        
        # 结果存储
        self.iteration_history: List[IterationResult] = []
        self.best_result: Optional[IterationResult] = None
        self.best_speedup = 0.0
        

    
    async def generate_kernel(self, level: int, problem_id: int) -> Dict[str, Any]:
        """
        生成kernel的主入口
        
        Args:
            level: KernelBench级别
            problem_id: 问题ID
            
        Returns:
            生成结果摘要
        """
        logger.info(f"开始生成kernel: Level {level} Problem {problem_id}")
        
        try:
            # 1. 加载问题数据
            db_loader = KernelBenchLoader()
            problem_info = db_loader.get_problem(level, problem_id)
            pytorch_forward, test_inputs, init_inputs = db_loader.execute_pytorch_code(problem_info)
            
            print(f"开始生成 {problem_info['operation_name']} kernel")
            print(f"输入形状: {problem_info['input_shapes']}")
            print(f"输出形状: {problem_info['output_shapes']}")
            
            # 2. 初始化迭代日志记录器
            from ..core.iteration_logger import IterationLogger
            iteration_logger = IterationLogger(
                output_dir=self.config.get("output", {}).get("base_dir", "generated_kernels"),
                level=level,
                problem_id=problem_id
            )
            
            # 3. 执行迭代生成
            for iteration in range(1, self.max_iterations + 1):
                print(f"\n第 {iteration}/{self.max_iterations} 轮迭代")
                
                iteration_result = await self._execute_iteration(
                    iteration, problem_info, pytorch_forward, test_inputs, init_inputs, iteration_logger
                )
                
                self.iteration_history.append(iteration_result)
                
                # 更新最佳结果
                if iteration_result.success and iteration_result.performance_metrics:
                    speedup = iteration_result.performance_metrics.get("speedup", 0.0)
                    if speedup > self.best_speedup:
                        self.best_speedup = speedup
                        self.best_result = iteration_result
                        print(f"新的最佳结果! 加速比: {speedup:.2f}x")
                
                # 记录迭代结果
                if iteration_result.success:
                    speedup = iteration_result.performance_metrics.get("speedup", 0.0) if iteration_result.performance_metrics else 0.0
                    print(f"迭代 {iteration} 完成: 正确性通过, 加速比 {speedup:.2f}x")
                else:
                    print(f"迭代 {iteration} 完成: 需要继续优化")
            
            # 4. 保存会话摘要
            session_summary_file = iteration_logger.save_session_summary()
            
            # 5. 生成最终摘要
            final_summary = self._generate_final_summary(problem_info)
            final_summary["session_summary_file"] = session_summary_file
            final_summary["best_kernel_path"] = iteration_logger.get_best_kernel_path()
            
            return final_summary
            
        except Exception as e:
            logger.error(f"生成过程失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "level": level,
                "problem_id": problem_id
            }
    
    async def _execute_iteration(self, 
                               iteration: int,
                               problem_info: Dict[str, Any],
                               pytorch_forward,
                               test_inputs,
                               init_inputs,
                               iteration_logger) -> IterationResult:
        """执行单次迭代"""
        
        result = IterationResult(iteration=iteration)
        
        # 记录迭代开始
        iteration_id = iteration_logger.log_iteration_start(
            iteration, "LangChainMultiAgent", {
                "problem_info": problem_info['operation_name'],
                "input_shapes": problem_info['input_shapes'],
                "iteration_type": "initial_analysis" if iteration == 1 else "debug_analysis"
            }
        )
        
        try:
            # 1. 分析阶段
            print("分析阶段...")
            
            if iteration == 1:
                # 首次迭代：分析PyTorch代码
                analysis_result = await self.analysis_chain.analyze_operation(
                    pytorch_code=problem_info["pytorch_code"],
                    problem_info=problem_info,
                    iteration=1
                )
            else:
                # 后续迭代：智能错误分析
                previous_results = self._prepare_previous_results()
                error_info = previous_results.get("error_info", "")
                previous_code = self._get_previous_code()
                performance_data = previous_results.get("performance_info", {})
                
                # 使用智能错误分析
                analysis_result = await self.analysis_chain.analyze_error_intelligently(
                    error_info=error_info,
                    code_context=previous_code,
                    performance_data=performance_data
                )
            
            result.analysis_result = analysis_result
            
            if not analysis_result.get("success", False):
                result.error_message = f"分析阶段失败: {analysis_result.get('error', '')}"
                return result
            
            # 2. 生成阶段
            print("代码生成阶段...")
            
            analysis_data = analysis_result["result"]
            
            if iteration == 1:
                # 首次迭代：根据架构设计生成代码
                generation_result = await self.generation_chain.generate_code(
                    pytorch_code=problem_info["pytorch_code"],
                    problem_info=problem_info,
                    architecture_design=analysis_data.get("kernel_structure", {}),
                    implementation_guidance=analysis_data.get("implementation_guidance", {})
                )
            else:
                # 后续迭代：基于智能分析修复代码
                previous_code = self._get_previous_code()
                
                # 使用智能分析结果进行代码修复
                generation_result = await self.generation_chain.fix_code_with_intelligent_analysis(
                    pytorch_code=problem_info["pytorch_code"],
                    current_code=previous_code,
                    error_analysis=analysis_data
                )
            
            result.generation_result = generation_result
            
            if not generation_result.get("success", False):
                result.error_message = f"代码生成阶段失败: {generation_result.get('error', '')}"
                return result
            
            # 3. 性能测试阶段
            print("性能测试阶段...")
            
            generated_code = generation_result["result"]["kernel_code"]
            result.final_code = generated_code
            
            # 保存生成的kernel代码
            if generated_code:
                kernel_path = iteration_logger.log_generated_kernel(
                    iteration, generated_code, 
                    f"{problem_info['operation_name']}_kernel", 
                    "LangChainGenerator"
                )
                print(f"代码已保存: {kernel_path}")
            
            # 执行性能测试
            performance_result = await self._run_performance_test(
                generated_code, pytorch_forward, test_inputs, init_inputs
            )
            
            result.performance_metrics = performance_result
            
            # 记录性能结果
            if performance_result:
                # # 准备额外的指标信息，包括错误详情
                # additional_metrics = {}
                # if "error" in performance_result:
                #     additional_metrics["error"] = performance_result["error"]
                # if "detailed_error" in performance_result:
                #     additional_metrics["detailed_error"] = performance_result["detailed_error"]
                # if "max_diff" in performance_result:
                #     additional_metrics["max_diff"] = performance_result["max_diff"]
                
                iteration_logger.log_performance_result(
                    iteration,
                    performance_result.get("triton_time", 0),
                    performance_result.get("pytorch_time", 0),
                    performance_result.get("speedup", 0),
                    performance_result.get("correctness", False),
                    # additional_metrics
                )
            
            # 4. 验证阶段
            print("验证阶段...")
            
            validation_result = await self.validation_chain.validate_kernel(
                pytorch_code=problem_info["pytorch_code"],
                triton_code=generated_code,
                test_results=performance_result,
                error_info=performance_result.get("error", ""),
                performance_data=performance_result
            )
            
            result.validation_result = validation_result
            
            # 基于正确性判断成功
            correctness = performance_result.get("correctness", False)
            speedup = performance_result.get("speedup", 0.0)
            
            if correctness:
                result.success = True
                print(f"成功! 正确性通过, 加速比: {speedup:.2f}x")
                
                # 如果性能不佳，进行性能优化分析
                if speedup < self.early_stop_threshold:
                    print(f"性能分析: 加速比{speedup:.2f}x低于目标{self.early_stop_threshold}x")
                    perf_analysis = await self.validation_chain.analyze_performance_optimization(
                        current_code=generated_code,
                        performance_metrics=performance_result,
                        target_performance={"target_speedup": f"{self.early_stop_threshold}x"}
                    )
                    if perf_analysis.get("success"):
                        result.performance_optimization = perf_analysis["result"]
            else:
                result.success = False
                print(f"失败: 正确性检查未通过")
            
            # 记录迭代完成
            detailed_output = {
                "kernel_generated": bool(result.final_code),
                "performance_tested": bool(result.performance_metrics),
                "analysis_success": result.analysis_result.get("success", False) if result.analysis_result else False,
                "generation_success": result.generation_result.get("success", False) if result.generation_result else False,
                "validation_success": result.validation_result.get("success", False) if result.validation_result else False,
                "performance_summary": {
                    "correctness": performance_result.get("correctness", False),
                    "speedup": performance_result.get("speedup", 0.0),
                    "has_error": "error" in performance_result,
                    "error_message": performance_result.get("error", "") if performance_result else ""
                } if performance_result else {}
            }
            
            # 构建完整的错误消息
            complete_error_message = result.error_message
            if performance_result and "error" in performance_result:
                if complete_error_message:
                    complete_error_message += f"; 性能测试错误: {performance_result['error']}"
                else:
                    complete_error_message = f"性能测试错误: {performance_result['error']}"
            
            iteration_logger.log_iteration_complete(
                iteration_id,
                detailed_output,
                result.success,
                complete_error_message,
                result.performance_metrics
            )
            
            return result
            
        except Exception as e:
            result.error_message = f"迭代执行失败: {str(e)}"
            logger.error(f"迭代 {iteration} 执行失败: {e}")
            
            # 记录失败的迭代
            iteration_logger.log_iteration_complete(
                iteration_id,
                {"exception_occurred": True},
                False,
                result.error_message,
                result.performance_metrics
            )
            
            return result
    
    async def _run_performance_test(self, kernel_code: str, pytorch_forward, 
                                  test_inputs, init_inputs) -> Dict[str, Any]:
        """运行性能测试"""
        try:
            import json
            
            # 直接调用性能测试逻辑，避免JSON序列化问题
            result = self.performance_tool._run_performance_test(
                kernel_code, pytorch_forward, test_inputs, init_inputs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "correctness": False,
                "speedup": 0.0
            }
    
    def _prepare_previous_results(self) -> Dict[str, Any]:
        """准备上次迭代的结果用于调试分析"""
        if not self.iteration_history:
            return {}
        
        last_iteration = self.iteration_history[-1]
        
        # 构建详细的历史信息，包含所有Agent的执行结果
        previous_results = {
            "generated_code": last_iteration.final_code or "",
            "error_info": self._get_previous_errors(),
            "performance_info": last_iteration.performance_metrics or {},
            "previous_design": last_iteration.analysis_result.get("result", {}) if last_iteration.analysis_result else {}
        }
        
        # 添加各个阶段的详细结果
        if last_iteration.analysis_result:
            previous_results["analysis_details"] = {
                "success": last_iteration.analysis_result.get("success", False),
                "result": last_iteration.analysis_result.get("result", {}),
                "error": last_iteration.analysis_result.get("error", "")
            }
        
        if last_iteration.generation_result:
            previous_results["generation_details"] = {
                "success": last_iteration.generation_result.get("success", False),
                "result": last_iteration.generation_result.get("result", {}),
                "error": last_iteration.generation_result.get("error", "")
            }
        
        if last_iteration.validation_result:
            previous_results["validation_details"] = {
                "success": last_iteration.validation_result.get("success", False),
                "result": last_iteration.validation_result.get("result", {}),
                "error": last_iteration.validation_result.get("error", "")
            }
        
        return previous_results
    
    def _get_previous_code(self) -> str:
        """获取上次生成的代码"""
        if not self.iteration_history:
            return ""
        
        last_iteration = self.iteration_history[-1]
        return last_iteration.final_code or ""
    
    def _get_previous_errors(self) -> str:
        """获取上次的错误信息"""
        if not self.iteration_history:
            return ""
        
        last_iteration = self.iteration_history[-1]
        errors = []
        
        if last_iteration.error_message:
            errors.append(f"迭代错误: {last_iteration.error_message}")
        
        if last_iteration.performance_metrics:
            if "error" in last_iteration.performance_metrics:
                error_msg = last_iteration.performance_metrics['error']
                
                # 结构化错误信息，提取关键的Triton错误
                if "program_id axis must be 0, 1, or 2" in error_msg:
                    errors.append("TRITON_GRID_ERROR: 使用了超过3维的网格，Triton只支持axis=0,1,2")
                elif "Mask argument cannot be block type" in error_msg:
                    errors.append("TRITON_MASK_ERROR: tl.load()中mask参数类型不匹配，需要确保pointer和mask维度一致")
                elif "shape mismatch" in error_msg.lower():
                    errors.append("SHAPE_MISMATCH_ERROR: 张量形状不匹配，检查索引计算")
                else:
                    errors.append(f"COMPILATION_ERROR: {error_msg}")
            
            # 添加更详细的错误信息
            if not last_iteration.performance_metrics.get("correctness", False):
                errors.append("CORRECTNESS_ERROR: Triton kernel输出与PyTorch不匹配")
            
            if last_iteration.performance_metrics.get("speedup", 0) <= 0:
                errors.append("PERFORMANCE_ERROR: 没有获得加速比或性能测试失败")
        
        # 添加分析、生成、验证阶段的错误
        if last_iteration.analysis_result and not last_iteration.analysis_result.get("success", False):
            errors.append(f"分析阶段错误: {last_iteration.analysis_result.get('error', '未知分析错误')}")
        
        if last_iteration.generation_result and not last_iteration.generation_result.get("success", False):
            errors.append(f"生成阶段错误: {last_iteration.generation_result.get('error', '未知生成错误')}")
        
        if last_iteration.validation_result and not last_iteration.validation_result.get("success", False):
            errors.append(f"验证阶段错误: {last_iteration.validation_result.get('error', '未知验证错误')}")
        
        return "\n".join(errors) if errors else "无错误信息"
    
    def _extract_fix_guidance(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """从分析结果中提取修复指导"""
        if analysis_data.get("analysis_type") == "debug_analysis":
            return {
                "fix_strategy": analysis_data.get("fix_strategy", {}),
                "updated_guidance": analysis_data.get("updated_guidance", {})
            }
        
        return {}
    
    def _generate_final_summary(self, problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终摘要"""
        total_iterations = len(self.iteration_history)
        successful_iterations = sum(1 for r in self.iteration_history if r.success)
        
        summary = {
            "success": self.best_result is not None,
            "level": problem_info.get("level", 0),
            "problem_id": problem_info.get("problem_id", 0),
            "operation_name": problem_info.get("operation_name", ""),
            "total_iterations": total_iterations,
            "successful_iterations": successful_iterations,
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_result.final_code if self.best_result else None
        }
        
        if self.best_result:
            summary["best_iteration"] = self.best_result.iteration
            summary["best_performance"] = self.best_result.performance_metrics
        
        # 添加迭代历史摘要
        summary["iteration_summary"] = []
        for i, result in enumerate(self.iteration_history, 1):
            iter_summary = {
                "iteration": i,
                "success": result.success,
                "speedup": result.performance_metrics.get("speedup", 0.0) if result.performance_metrics else 0.0,
                "error": result.error_message if result.error_message else None
            }
            summary["iteration_summary"].append(iter_summary)
        
        return summary