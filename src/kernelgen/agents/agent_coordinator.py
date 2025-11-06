"""
Agent Coordinator - 重新设计的三Agent协作系统
基于aikg的最佳实践，实现正确的Agent协作流程
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .analyzer_agent import AnalyzerAgent
from .generator_agent import GeneratorAgent
from .validator_agent import ValidatorAgent
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

class AgentCoordinator:
    """
    重新设计的Agent协调器
    
    正确的工作流程：
    1. 首次迭代：AnalyzerAgent分析PyTorch代码 → GeneratorAgent生成代码 → ValidatorAgent测试
    2. 后续迭代：AnalyzerAgent调试分析 → GeneratorAgent修复代码 → ValidatorAgent测试
    3. ValidatorAgent提供专门的反馈给AnalyzerAgent和GeneratorAgent
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        """
        初始化协调器
        
        Args:
            llm_client: LLM客户端
            config: 配置信息
        """
        self.llm_client = llm_client
        self.config = config
        
        # 初始化三个Agent
        self.analyzer_agent = AnalyzerAgent(llm_client, config)
        self.generator_agent = GeneratorAgent(llm_client, config)
        self.validator_agent = ValidatorAgent(llm_client, config)
        
        # 初始化性能测试器
        performance_config = config.get("performance", {})
        self.benchmark = TritonPerformanceBenchmark(
            device=performance_config.get("device", "cuda"),
            warmup_runs=performance_config.get("warmup_runs", 10),
            benchmark_runs=performance_config.get("benchmark_runs", 100)
        )
        
        # 迭代控制参数
        self.max_iterations = config.get("generation", {}).get("max_iterations", 5)
        self.early_stop_threshold = config.get("generation", {}).get("early_stop_threshold", 1.2)
        
        # 结果存储
        self.iteration_history: List[IterationResult] = []
        self.best_result: Optional[IterationResult] = None
        self.best_speedup = 0.0
        
        logger.info("Agent协调器初始化完成")
    
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
            
            print(f"🎯 开始生成 {problem_info['operation_name']} kernel")
            print(f"   输入形状: {problem_info['input_shapes']}")
            print(f"   输出形状: {problem_info['output_shapes']}")
            
            # 2. 初始化迭代日志记录器
            from ..core.iteration_logger import IterationLogger
            iteration_logger = IterationLogger(
                output_dir=self.config.get("output", {}).get("base_dir", "generated_kernels"),
                level=level,
                problem_id=problem_id
            )
            
            # 3. 执行迭代生成
            for iteration in range(1, self.max_iterations + 1):
                print(f"\n🔄 第 {iteration}/{self.max_iterations} 轮迭代")
                
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
                        print(f"   🎉 新的最佳结果! 加速比: {speedup:.2f}x")
                
                # 记录迭代结果，但不进行早停判断
                if iteration_result.success:
                    speedup = iteration_result.performance_metrics.get("speedup", 0.0) if iteration_result.performance_metrics else 0.0
                    print(f"   📈 迭代 {iteration} 完成: 正确性通过, 加速比 {speedup:.2f}x")
                else:
                    print(f"   📝 迭代 {iteration} 完成: 需要继续优化")
                
                # 显示ValidatorAgent的评估（仅作信息参考）
                if iteration_result.validation_result and iteration_result.validation_result.get("success", False):
                    validation_data = iteration_result.validation_result["result"]
                    validation_status = validation_data.get("validation_result", {}).get("status", "error")
                    print(f"      ValidatorAgent评估: {validation_status}")
            
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
            iteration, "MultiAgent", {
                "problem_info": problem_info['operation_name'],
                "input_shapes": problem_info['input_shapes'],
                "iteration_type": "initial_analysis" if iteration == 1 else "debug_analysis"
            }
        )
        
        try:
            # 1. AnalyzerAgent阶段
            print("   🔍 分析阶段...")
            
            if iteration == 1:
                # 首次迭代：分析PyTorch代码
                analysis_result = await self.analyzer_agent.analyze_operation(
                    pytorch_code=problem_info["pytorch_code"],
                    problem_info=problem_info,
                    iteration=1
                )
            else:
                # 后续迭代：调试分析
                previous_results = self._prepare_previous_results()
                analysis_result = await self.analyzer_agent.analyze_operation(
                    pytorch_code=problem_info["pytorch_code"],
                    problem_info=problem_info,
                    iteration=iteration,
                    previous_results=previous_results
                )
            
            result.analysis_result = analysis_result
            
            if not analysis_result.get("success", False):
                result.error_message = f"分析阶段失败: {analysis_result.get('error', '')}"
                return result
            
            # 2. GeneratorAgent阶段
            print("   💻 代码生成阶段...")
            
            analysis_data = analysis_result["result"]
            
            if iteration == 1:
                # 首次迭代：根据架构设计生成代码
                generation_result = await self.generator_agent.generate_code(
                    pytorch_code=problem_info["pytorch_code"],
                    problem_info=problem_info,
                    architecture_design=analysis_data.get("architecture_design", {}),
                    implementation_guidance=analysis_data.get("implementation_guidance", {})
                )
            else:
                # 后续迭代：修复代码
                previous_code = self._get_previous_code()
                previous_errors = self._get_previous_errors()
                fix_guidance = self._extract_fix_guidance(analysis_data)
                
                generation_result = await self.generator_agent.fix_code(
                    pytorch_code=problem_info["pytorch_code"],
                    current_code=previous_code,
                    error_info=previous_errors,
                    fix_guidance=fix_guidance
                )
            
            result.generation_result = generation_result
            
            if not generation_result.get("success", False):
                result.error_message = f"代码生成阶段失败: {generation_result.get('error', '')}"
                return result
            
            # 3. 性能测试阶段
            print("   ⚡ 性能测试阶段...")
            
            generated_code = generation_result["result"]["kernel_code"]
            result.final_code = generated_code
            
            # 保存生成的kernel代码
            if generated_code:
                kernel_path = iteration_logger.log_generated_kernel(
                    iteration, generated_code, 
                    f"{problem_info['operation_name']}_kernel", 
                    "GeneratorAgent"
                )
                print(f"      代码已保存: {kernel_path}")
            
            # 执行性能测试
            performance_result = await self._run_performance_test(
                generated_code, pytorch_forward, test_inputs, init_inputs
            )
            
            result.performance_metrics = performance_result
            
            # 记录性能结果
            if performance_result:
                iteration_logger.log_performance_result(
                    iteration,
                    performance_result.get("triton_time", 0),
                    performance_result.get("pytorch_time", 0),
                    performance_result.get("speedup", 0),
                    performance_result.get("correctness", False)
                )
            
            # 4. ValidatorAgent阶段
            print("   ✅ 验证阶段...")
            
            validation_result = await self.validator_agent.validate_kernel(
                pytorch_code=problem_info["pytorch_code"],
                triton_code=generated_code,
                test_results=performance_result,
                error_info=performance_result.get("error", ""),
                performance_data=performance_result
            )
            
            result.validation_result = validation_result
            
            # 基于正确性判断成功，而不是基于ValidatorAgent的复杂判断
            correctness = performance_result.get("correctness", False)
            speedup = performance_result.get("speedup", 0.0)
            
            if correctness:
                result.success = True
                print(f"   🎉 成功! 正确性通过, 加速比: {speedup:.2f}x")
                if speedup >= 1.0:
                    print(f"      ✨ 性能优于PyTorch!")
                elif speedup >= 0.8:
                    print(f"      ✅ 性能接近PyTorch")
                else:
                    print(f"      📊 性能低于PyTorch，但功能正确")
            else:
                result.success = False
                print(f"   ❌ 失败: 正确性检查未通过")
            
            # 仍然记录ValidatorAgent的分析结果，但不用于成功判断
            if validation_result.get("success", False):
                validation_data = validation_result["result"]
                validation_status = validation_data.get("validation_result", {}).get("status", "error")
                print(f"      ValidatorAgent评估: {validation_status}")
            else:
                print(f"      ValidatorAgent分析失败: {validation_result.get('error', '')}")
            
            # 记录迭代完成 - 增强版日志
            detailed_output = {
                "kernel_generated": bool(result.final_code),
                "performance_tested": bool(result.performance_metrics),
                "analysis_success": result.analysis_result.get("success", False) if result.analysis_result else False,
                "generation_success": result.generation_result.get("success", False) if result.generation_result else False,
                "validation_success": result.validation_result.get("success", False) if result.validation_result else False,
                
                # 详细的Agent结果
                "analysis_result": {
                    "success": result.analysis_result.get("success", False) if result.analysis_result else False,
                    "error": result.analysis_result.get("error", "") if result.analysis_result else "",
                    "analysis_type": result.analysis_result.get("result", {}).get("analysis_type", "") if result.analysis_result else ""
                },
                "generation_result": {
                    "success": result.generation_result.get("success", False) if result.generation_result else False,
                    "error": result.generation_result.get("error", "") if result.generation_result else "",
                    "kernel_name": result.generation_result.get("result", {}).get("kernel_name", "") if result.generation_result else ""
                },
                "validation_result": {
                    "success": result.validation_result.get("success", False) if result.validation_result else False,
                    "error": result.validation_result.get("error", "") if result.validation_result else "",
                    "status": result.validation_result.get("result", {}).get("validation_result", {}).get("status", "") if result.validation_result else ""
                },
                
                # 错误信息汇总
                "all_errors": self._collect_all_errors(result),
                
                # 性能数据
                "performance_summary": {
                    "correctness": performance_result.get("correctness", False),
                    "speedup": performance_result.get("speedup", 0.0),
                    "has_error": "error" in performance_result
                } if performance_result else {}
            }
            
            iteration_logger.log_iteration_complete(
                iteration_id,
                detailed_output,
                result.success,
                result.error_message,
                result.performance_metrics
            )
            
            return result
            
        except Exception as e:
            result.error_message = f"迭代执行失败: {str(e)}"
            logger.error(f"迭代 {iteration} 执行失败: {e}")
            
            # 记录失败的迭代
            iteration_logger.log_iteration_complete(
                iteration_id,
                {
                    "kernel_generated": bool(result.final_code),
                    "performance_tested": bool(result.performance_metrics),
                    "exception_occurred": True
                },
                False,
                result.error_message,
                result.performance_metrics
            )
            
            return result
    
    async def _run_performance_test(self, 
                                  kernel_code: str,
                                  pytorch_forward,
                                  test_inputs,
                                  init_inputs) -> Dict[str, Any]:
        """运行性能测试"""
        try:
            # 1. 首先确保输入tensor在CUDA设备上（方案B：在所有测试前统一设备）
            import torch
            if torch.cuda.is_available():
                cuda_test_inputs = []
                for inp in test_inputs:
                    if isinstance(inp, torch.Tensor):
                        cuda_test_inputs.append(inp.cuda())
                    else:
                        cuda_test_inputs.append(inp)
                test_inputs = cuda_test_inputs
                
                print(f"      输入tensor已移动到CUDA: {[inp.device for inp in test_inputs if isinstance(inp, torch.Tensor)]}")
            else:
                return {
                    "success": False,
                    "error": "CUDA不可用，无法运行Triton kernel",
                    "correctness": False,
                    "speedup": 0.0
                }
            
            # 2. 运行PyTorch基准测试（现在使用CUDA输入）
            pytorch_time = None
            pytorch_result = None
            
            try:
                # 获取PyTorch参考结果和性能（使用CUDA输入）
                if init_inputs and isinstance(init_inputs, dict) and init_inputs:
                    pytorch_result = pytorch_forward(*test_inputs, **init_inputs)
                elif init_inputs and isinstance(init_inputs, list) and init_inputs:
                    pytorch_result = pytorch_forward(*test_inputs, *init_inputs)
                else:
                    pytorch_result = pytorch_forward(*test_inputs)
                
                # 测试PyTorch性能（使用CUDA输入）
                pytorch_time = self.benchmark._benchmark_pytorch_general(pytorch_forward, test_inputs)
                print(f"      PyTorch基准: {pytorch_time:.4f}ms (CUDA)")
                
            except Exception as pytorch_error:
                print(f"      PyTorch基准测试失败: {pytorch_error}")
                return {
                    "success": False,
                    "error": f"PyTorch基准测试失败: {pytorch_error}",
                    "correctness": False,
                    "speedup": 0.0
                }
            
            # 3. 编译Triton kernel
            triton_func = self.benchmark.compile_and_load_kernel(kernel_code)
            if not triton_func:
                return {
                    "success": False,
                    "error": "Triton kernel编译失败",
                    "correctness": False,
                    "speedup": 0.0,
                    "pytorch_time": pytorch_time,  # 至少返回PyTorch数据
                    "triton_time": 0.0
                }
            
            # 4. 正确性测试（现在两个结果都在CUDA上）
            try:
                # 获取Triton结果（使用CUDA输入）
                triton_result = triton_func(*test_inputs)
                
                # 正确性检查（现在两个结果都在同一设备上）
                if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                    max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
                    correctness = max_diff < 1e-4
                    
                    print(f"      正确性检查: {'通过' if correctness else '失败'}, 最大差异: {max_diff:.6f}")
                    print(f"      PyTorch结果范围: [{pytorch_result.min().item():.6f}, {pytorch_result.max().item():.6f}] (CUDA)")
                    print(f"      Triton结果范围: [{triton_result.min().item():.6f}, {triton_result.max().item():.6f}] (CUDA)")
                    print(f"      设备检查: PyTorch={pytorch_result.device}, Triton={triton_result.device}")
                else:
                    correctness = False
                    max_diff = float('inf')
                    print(f"      正确性检查失败: 结果类型不匹配")
                
            except Exception as correctness_error:
                print(f"      正确性检查失败: {correctness_error}")
                return {
                    "success": False,
                    "error": f"正确性检查失败: {correctness_error}",
                    "correctness": False,
                    "speedup": 0.0,
                    "pytorch_time": pytorch_time,  # 至少返回PyTorch数据
                    "triton_time": 0.0
                }
            
            # 5. 性能测试
            if correctness:
                try:
                    # 使用现有的benchmark_general方法 - 专门为Triton设计
                    perf_result = self.benchmark.benchmark_general(
                        triton_func, pytorch_forward, test_inputs
                    )
                    
                    if perf_result.get("success", False):
                        speedup = perf_result.get("speedup", 0.0)
                        pytorch_time = perf_result.get("pytorch_time_ms", 0.0)
                        triton_time = perf_result.get("triton_time_ms", 0.0)
                        
                        print(f"      性能测试: PyTorch {pytorch_time:.4f}ms, Triton {triton_time:.4f}ms, 加速比 {speedup:.2f}x")
                        
                        return {
                            "success": True,
                            "correctness": True,
                            "speedup": speedup,
                            "pytorch_time": pytorch_time,
                            "triton_time": triton_time,
                            "max_diff": max_diff
                        }
                    else:
                        print(f"      性能测试失败: {perf_result.get('error', '未知错误')}")
                        return {
                            "success": True,  # 正确性通过了
                            "correctness": True,
                            "speedup": 0.0,
                            "error": f"性能测试失败: {perf_result.get('error', '未知错误')}",
                            "max_diff": max_diff
                        }
                    
                except Exception as perf_error:
                    print(f"      性能测试失败: {perf_error}")
                    return {
                        "success": True,  # 正确性通过了
                        "correctness": True,
                        "speedup": 0.0,
                        "error": f"性能测试失败: {perf_error}",
                        "max_diff": max_diff
                    }
            else:
                return {
                    "success": False,
                    "correctness": False,
                    "speedup": 0.0,
                    "max_diff": max_diff,
                    "error": "正确性检查未通过",
                    "pytorch_time": pytorch_time,  # 至少返回PyTorch数据
                    "triton_time": 0.0
                }
                
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
        
        return {
            "generated_code": last_iteration.final_code or "",
            "error_info": self._get_previous_errors(),  # 使用完整的错误信息收集方法
            "performance_info": last_iteration.performance_metrics or {},
            "previous_design": last_iteration.analysis_result.get("result", {}) if last_iteration.analysis_result else {}
        }
    
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
        
        if last_iteration.performance_metrics and "error" in last_iteration.performance_metrics:
            errors.append(f"性能测试错误: {last_iteration.performance_metrics['error']}")
        
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
            "final_code": self.best_result.final_code if self.best_result else None
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
    
    def _collect_all_errors(self, result: IterationResult) -> List[str]:
        """收集迭代中的所有错误信息"""
        errors = []
        
        # 迭代级别错误
        if result.error_message:
            errors.append(f"迭代错误: {result.error_message}")
        
        # Agent级别错误
        if result.analysis_result and not result.analysis_result.get("success", False):
            errors.append(f"分析错误: {result.analysis_result.get('error', '')}")
        
        if result.generation_result and not result.generation_result.get("success", False):
            errors.append(f"生成错误: {result.generation_result.get('error', '')}")
        
        if result.validation_result and not result.validation_result.get("success", False):
            errors.append(f"验证错误: {result.validation_result.get('error', '')}")
        
        # 性能测试错误
        if result.performance_metrics and "error" in result.performance_metrics:
            errors.append(f"性能测试错误: {result.performance_metrics['error']}")
        
        return errors