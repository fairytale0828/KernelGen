"""
Agent Coordinator - 四个Agent的协调器
负责管理四个Agent之间的协作流程
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from dataclasses import dataclass

from .base_agent import BaseAgent
from .design_agent import DesignAgent
from .code_agent import CodeAgent
from .optimize_agent import OptimizeAgent
from .debug_agent import DebugAgent
from ..core.performance_benchmark import TritonPerformanceBenchmark
from ..dataset import KernelBenchLoader

logger = logging.getLogger(__name__)

@dataclass
class IterationResult:
    """单次迭代结果"""
    iteration: int
    design_result: Optional[Dict[str, Any]] = None
    code_result: Optional[Dict[str, Any]] = None
    performance_result: Optional[Dict[str, Any]] = None
    optimize_result: Optional[Dict[str, Any]] = None
    debug_result: Optional[Dict[str, Any]] = None
    final_code: Optional[str] = None
    success: bool = False
    error_message: str = ""

class AgentCoordinator:
    """
    Agent协调器 - 管理四个Agent的协作
    
    工作流程：
    1. Design Agent 设计kernel架构
    2. Code Agent 生成具体代码
    3. 性能测试和错误检测
    4. 如果有错误 -> Debug Agent 分析 -> 回到Design Agent
    5. 如果性能不佳 -> Optimize Agent 建议 -> 回到Design Agent
    6. 迭代直到成功或达到最大次数
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
        
        # 初始化四个Agent
        self.design_agent = DesignAgent(llm_client, config)
        self.code_agent = CodeAgent(llm_client, config)
        self.optimize_agent = OptimizeAgent(llm_client, config)
        self.debug_agent = DebugAgent(llm_client, config)
        
        # 初始化性能测试器
        performance_config = config.get("performance", {})
        self.benchmark = TritonPerformanceBenchmark(
            device=performance_config.get("device", "cuda"),
            warmup_runs=performance_config.get("warmup_runs", 10),
            benchmark_runs=performance_config.get("benchmark_runs", 100)
        )
        
        # 迭代控制参数
        self.max_iterations = config.get("generation", {}).get("max_iterations", 10)
        self.early_stop_threshold = config.get("generation", {}).get("early_stop_threshold", 1.2)
        self.min_successful_iterations = config.get("generation", {}).get("min_successful_iterations", 2)
        
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
            # 1. 加载问题
            problem_info = self._load_problem(level, problem_id)
            pytorch_code = problem_info["code"]
            # print(pytorch_code)
            
            # 2. 准备PyTorch参考函数和测试输入
            pytorch_func, test_inputs = self._prepare_reference(pytorch_code)
            # print("-----------------------------------------------------------")
            # print(pytorch_func)
            if not pytorch_func or not test_inputs:
                return self._create_failure_summary("无法创建PyTorch参考函数或测试输入")
            
            # 3. 迭代生成和优化
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"🔄 第{iteration}轮迭代")
                
                result = await self._run_iteration(
                    iteration, problem_info, pytorch_func, test_inputs
                )
                
                self.iteration_history.append(result)
                
                if result.success:
                    self._update_best_result(result)
                    
                    # 检查早停条件
                    if self._should_early_stop(result):
                        logger.info(f"🎯 达到早停条件，迭代结束")
                        break
                
                # 显示进度
                self._log_progress(iteration)
            
            # 4. 返回结果摘要
            return self._create_success_summary(problem_info)
            
        except Exception as e:
            logger.error(f"生成kernel失败: {str(e)}")
            return self._create_failure_summary(str(e))
    
    def _load_problem(self, level: int, problem_id: int) -> Dict[str, Any]:
        """加载KernelBench问题"""
        
        dataset_config = {
            "source": "huggingface",
            "name": "ScalingIntelligence/KernelBench",
            "level": level,
            "problem_ids": [problem_id]
        }
        
        loader = KernelBenchLoader(dataset_config)
        loader.load_dataset()
        
        return loader.get_problem_by_id(problem_id)
    
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
    
    async def _run_iteration(self, iteration: int, problem_info: Dict[str, Any],
                           pytorch_func: Any, test_inputs: List[Any]) -> IterationResult:
        """运行单次迭代"""
        
        result = IterationResult(iteration=iteration)
        
        try:
            # 1. Design Agent 设计
            design_result = await self._run_design_agent(iteration, problem_info)
            result.design_result = design_result
            
            if not design_result["success"]:
                result.error_message = f"设计阶段失败: {design_result['error_message']}"
                return result
            
            # 2. Code Agent 生成代码
            code_result = await self._run_code_agent(design_result, problem_info)
            result.code_result = code_result
            
            if not code_result["success"]:
                result.error_message = f"代码生成失败: {code_result['error_message']}"
                return result
            
            generated_code = code_result["content"].get("generated_code", "")
            result.final_code = generated_code
            
            # 3. 性能测试和错误检测
            performance_result = self._test_kernel_performance(
                generated_code, pytorch_func, test_inputs
            )
            result.performance_result = performance_result
            
            # 4. 根据测试结果决定下一步
            if performance_result.get("has_error", False):
                # 有错误，运行Debug Agent
                debug_result = await self._run_debug_agent(performance_result, generated_code, problem_info["code"])
                result.debug_result = debug_result
                
                # 将debug结果反馈给下一轮的Design Agent
                self._store_debug_feedback(debug_result)
                
            elif performance_result.get("speedup", 0) < self.early_stop_threshold:
                # 性能不佳，运行Optimize Agent
                optimize_result = await self._run_optimize_agent(performance_result, generated_code)
                result.optimize_result = optimize_result
                
                # 将优化建议反馈给下一轮的Design Agent
                self._store_optimize_feedback(optimize_result)
                
            else:
                # 成功
                result.success = True
            
        except Exception as e:
            result.error_message = f"迭代执行异常: {str(e)}"
            logger.error(f"第{iteration}轮迭代失败: {str(e)}")
        
        return result
    
    async def _run_design_agent(self, iteration: int, problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """运行Design Agent"""
        
        # 构造task_info
        task_info = {
            "pytorch_code": problem_info["code"],
            "problem_info": problem_info
        }
        
        if iteration > 1:
            # 基于反馈的重新设计
            feedback = self._get_latest_feedback()
            
            if feedback.get("type") == "debug":
                task_info["debug_feedback"] = feedback.get("detailed_guidance", "")
            else:  # optimize feedback
                task_info["optimization_feedback"] = feedback.get("detailed_analysis", "")
        
        # 调用Design Agent
        try:
            result_text, _, _ = await self.design_agent.run(task_info)
            
            # 解析JSON结果
            import json
            result_data = json.loads(result_text)
            
            return {
                "success": True,
                "content": result_data,
                "error_message": ""
            }
            
        except Exception as e:
            logger.error(f"Design Agent执行失败: {str(e)}")
            return {
                "success": False,
                "content": {},
                "error_message": str(e)
            }
    
    async def _run_code_agent(self, design_result: Dict[str, Any], problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """运行Code Agent"""
        
        design_content = design_result.get("content", {})
        
        # 构造task_info
        task_info = {
            "design_plan": design_content.get("design_plan", {}),
            "implementation_guide": design_content.get("implementation_guide", {}),
            "pytorch_code": problem_info["code"],
            "operator_analysis": design_content.get("operator_analysis", {})
        }
        
        # 如果有错误反馈，添加到task_info
        feedback = self._get_latest_feedback()
        if feedback.get("type") == "debug":
            task_info["error_feedback"] = feedback.get("detailed_guidance", "")
        
        # 调用Code Agent
        try:
            result_text, _, _ = await self.code_agent.run(task_info)
            
            # 解析JSON结果
            import json
            result_data = json.loads(result_text)
            
            return {
                "success": True,
                "content": result_data,
                "error_message": ""
            }
            
        except Exception as e:
            logger.error(f"Code Agent执行失败: {str(e)}")
            return {
                "success": False,
                "content": {},
                "error_message": str(e)
            }
    
    async def _run_optimize_agent(self, performance_result: Dict[str, Any], 
                                generated_code: str) -> Dict[str, Any]:
        """运行Optimize Agent"""
        
        # 构造task_info
        task_info = {
            "performance_data": performance_result,
            "kernel_code": generated_code
        }
        
        # 调用Optimize Agent
        try:
            result_text, _, _ = await self.optimize_agent.run(task_info)
            
            # 解析JSON结果
            import json
            result_data = json.loads(result_text)
            
            return {
                "success": True,
                "content": result_data,
                "error_message": ""
            }
            
        except Exception as e:
            logger.error(f"Optimize Agent执行失败: {str(e)}")
            return {
                "success": False,
                "content": {},
                "error_message": str(e)
            }
    
    async def _run_debug_agent(self, performance_result: Dict[str, Any], 
                             failed_code: str, pytorch_code: str) -> Dict[str, Any]:
        """运行Debug Agent"""
        
        # 构造task_info
        task_info = {
            "error_info": performance_result.get("error_info", {}),
            "failed_code": failed_code,
            "pytorch_code": pytorch_code
        }
        
        # 调用Debug Agent
        try:
            result_text, _, _ = await self.debug_agent.run(task_info)
            
            # 解析JSON结果
            import json
            result_data = json.loads(result_text)
            
            return {
                "success": True,
                "content": result_data,
                "error_message": ""
            }
            
        except Exception as e:
            logger.error(f"Debug Agent执行失败: {str(e)}")
            return {
                "success": False,
                "content": {},
                "error_message": str(e)
            }
    
    def _test_kernel_performance(self, kernel_code: str, pytorch_func: Any, 
                               test_inputs: List[Any]) -> Dict[str, Any]:
        """测试kernel性能"""
        
        result = {
            "has_error": False,
            "error_info": {},
            "speedup": 0.0,
            "triton_time_ms": 0.0,
            "pytorch_time_ms": 0.0,
            "success": False
        }
        
        try:
            # 1. 语法验证
            is_valid, error_msg = self.benchmark.validate_kernel_syntax(kernel_code)
            if not is_valid:
                result["has_error"] = True
                result["error_info"] = {
                    "error_type": "compilation_error",
                    "error_message": error_msg
                }
                return result
            
            # 2. 编译测试
            compiled_func = self.benchmark.compile_and_load_kernel(kernel_code, "test_kernel")
            if not compiled_func:
                result["has_error"] = True
                result["error_info"] = {
                    "error_type": "compilation_error", 
                    "error_message": "编译失败"
                }
                return result
            
            # 3. 运行时测试
            try:
                triton_result = compiled_func(*test_inputs)
            except Exception as e:
                result["has_error"] = True
                result["error_info"] = {
                    "error_type": "runtime_error",
                    "error_message": str(e)
                }
                return result
            
            # 4. 正确性验证
            try:
                import torch
                pytorch_result = pytorch_func(*test_inputs)
                
                if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                    is_correct = torch.allclose(pytorch_result, triton_result, rtol=1e-4, atol=1e-5)
                    if not is_correct:
                        result["has_error"] = True
                        result["error_info"] = {
                            "error_type": "correctness_error",
                            "error_message": "计算结果不正确"
                        }
                        return result
                else:
                    result["has_error"] = True
                    result["error_info"] = {
                        "error_type": "correctness_error",
                        "error_message": "结果类型不匹配"
                    }
                    return result
            
            except Exception as e:
                result["has_error"] = True
                result["error_info"] = {
                    "error_type": "correctness_error",
                    "error_message": f"正确性验证失败: {str(e)}"
                }
                return result
            
            # 5. 性能测试
            try:
                perf_results = self.benchmark.benchmark_general(compiled_func, pytorch_func, test_inputs)
                
                if perf_results["success"] and perf_results["speedups"]:
                    result["speedup"] = perf_results["speedups"][0]
                    result["triton_time_ms"] = perf_results["triton_times"][0]
                    result["pytorch_time_ms"] = perf_results["pytorch_times"][0]
                    result["success"] = True
                else:
                    result["has_error"] = True
                    result["error_info"] = {
                        "error_type": "performance_error",
                        "error_message": f"性能测试失败: {perf_results.get('error', '未知错误')}"
                    }
            
            except Exception as e:
                result["has_error"] = True
                result["error_info"] = {
                    "error_type": "performance_error",
                    "error_message": f"性能测试异常: {str(e)}"
                }
        
        except Exception as e:
            result["has_error"] = True
            result["error_info"] = {
                "error_type": "unknown_error",
                "error_message": f"测试过程异常: {str(e)}"
            }
        
        return result
    
    def _store_debug_feedback(self, debug_result: Dict[str, Any]):
        """存储debug反馈"""
        
        if hasattr(self, '_latest_feedback'):
            del self._latest_feedback
        
        error_analysis = debug_result.get("content", {}).get("error_analysis", {})
        fix_suggestions = error_analysis.get("fix_suggestions", {})
        
        self._latest_feedback = {
            "type": "debug",
            "error_info": error_analysis.get("root_cause", {}),
            "suggestions": [],
            "detailed_guidance": fix_suggestions.get("detailed_guidance", "")
        }
    
    def _store_optimize_feedback(self, optimize_result: Dict[str, Any]):
        """存储优化反馈"""
        
        if hasattr(self, '_latest_feedback'):
            del self._latest_feedback
        
        performance_analysis = optimize_result.get("content", {}).get("performance_analysis", {})
        optimization_suggestions = performance_analysis.get("optimization_suggestions", {})
        
        self._latest_feedback = {
            "type": "optimize",
            "performance_data": performance_analysis.get("bottleneck_analysis", {}),
            "suggestions": [],
            "detailed_analysis": optimization_suggestions.get("detailed_analysis", "")
        }
    
    def _get_latest_feedback(self) -> Dict[str, Any]:
        """获取最新反馈"""
        
        return getattr(self, '_latest_feedback', {})
    
    def _update_best_result(self, result: IterationResult):
        """更新最佳结果"""
        
        if not result.success or not result.performance_result:
            return
        
        current_speedup = result.performance_result.get("speedup", 0.0)
        
        if current_speedup > self.best_speedup:
            self.best_speedup = current_speedup
            self.best_result = result
            logger.info(f"🏆 发现更好的kernel: {current_speedup:.2f}x加速")
    
    def _should_early_stop(self, result: IterationResult) -> bool:
        """检查是否应该早停"""
        
        if not result.success or not result.performance_result:
            return False
        
        speedup = result.performance_result.get("speedup", 0.0)
        successful_iterations = len([r for r in self.iteration_history if r.success])
        
        return (speedup >= self.early_stop_threshold and 
                successful_iterations >= self.min_successful_iterations)
    
    def _log_progress(self, iteration: int):
        """记录进度"""
        
        total_iterations = len(self.iteration_history)
        successful_iterations = len([r for r in self.iteration_history if r.success])
        success_rate = successful_iterations / total_iterations * 100 if total_iterations > 0 else 0
        
        logger.info(f"📊 进度更新:")
        logger.info(f"   已完成: {iteration}/{self.max_iterations}")
        logger.info(f"   成功率: {successful_iterations}/{total_iterations} ({success_rate:.1f}%)")
        logger.info(f"   最佳加速比: {self.best_speedup:.2f}x")
    
    def _create_success_summary(self, problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """创建成功摘要"""
        
        total_iterations = len(self.iteration_history)
        successful_iterations = len([r for r in self.iteration_history if r.success])
        
        return {
            "success": True,
            "problem_info": problem_info,
            "total_iterations": total_iterations,
            "successful_iterations": successful_iterations,
            "success_rate": successful_iterations / total_iterations * 100 if total_iterations > 0 else 0,
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_result.final_code if self.best_result else None,
            "best_performance": self.best_result.performance_result if self.best_result else None,
            "iteration_history": [
                {
                    "iteration": r.iteration,
                    "success": r.success,
                    "speedup": r.performance_result.get("speedup", 0.0) if r.performance_result else 0.0,
                    "error_message": r.error_message
                }
                for r in self.iteration_history
            ]
        }
    
    def _create_failure_summary(self, error_message: str) -> Dict[str, Any]:
        """创建失败摘要"""
        
        return {
            "success": False,
            "error_message": error_message,
            "total_iterations": len(self.iteration_history),
            "successful_iterations": len([r for r in self.iteration_history if r.success]),
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_result.final_code if self.best_result else None,
            "iteration_history": []
        }