"""
性能测试相关的LangChain工具
"""

import json
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import BaseTool

from ..core.performance_benchmark import TritonPerformanceBenchmark

logger = logging.getLogger(__name__)

class PerformanceBenchmarkTool(BaseTool):
    """性能基准测试工具"""
    
    name: str = "performance_benchmark"
    description: str = """
    运行Triton kernel性能基准测试。
    输入: JSON格式的测试配置，包含kernel_code, test_inputs等
    输出: 性能测试结果，包含正确性和性能指标
    """
    
    def __init__(self, benchmark: TritonPerformanceBenchmark, **kwargs):
        super().__init__(**kwargs)
        self._benchmark = benchmark
    
    @property
    def benchmark(self):
        return self._benchmark
    
    def _run(self, tool_input: str) -> str:
        """运行性能测试"""
        try:
            # 解析输入
            config = json.loads(tool_input)
            kernel_code = config["kernel_code"]
            pytorch_forward = config["pytorch_forward"]
            test_inputs = config["test_inputs"]
            init_inputs = config.get("init_inputs")
            
            # 运行性能测试
            result = self._run_performance_test(
                kernel_code, pytorch_forward, test_inputs, init_inputs
            )
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"性能测试工具执行失败: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "correctness": False,
                "speedup": 0.0
            })
    
    def _run_performance_test(self, kernel_code: str, pytorch_forward, 
                            test_inputs, init_inputs) -> Dict[str, Any]:
        """执行性能测试逻辑"""
        try:
            # 确保输入在CUDA设备上
            import torch
            if torch.cuda.is_available():
                cuda_test_inputs = []
                for inp in test_inputs:
                    if isinstance(inp, torch.Tensor):
                        cuda_test_inputs.append(inp.cuda())
                    else:
                        cuda_test_inputs.append(inp)
                test_inputs = cuda_test_inputs
            else:
                return {
                    "success": False,
                    "error": "CUDA不可用",
                    "correctness": False,
                    "speedup": 0.0
                }
            
            # 运行PyTorch基准 - 使用KernelBench原始模型
            try:
                # pytorch_forward是从KernelBench数据库中执行的原始模型forward方法
                # 它已经使用get_init_inputs()正确初始化了模型参数
                # test_inputs来自get_inputs()，是正确的测试输入
                pytorch_result = pytorch_forward(*test_inputs)
                
                pytorch_time = self.benchmark._benchmark_pytorch_general(pytorch_forward, test_inputs)
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"PyTorch基准测试失败: {e}",
                    "correctness": False,
                    "speedup": 0.0
                }
            
            # 编译Triton kernel
            triton_func = self.benchmark.compile_and_load_kernel(kernel_code)
            if not triton_func:
                return {
                    "success": False,
                    "error": "Triton kernel编译失败",
                    "correctness": False,
                    "speedup": 0.0,
                    "pytorch_time": pytorch_time,
                    "triton_time": 0.0
                }
            
            # 正确性测试
            additional_args = []  # 初始化额外参数列表
            try:
                # 检查Triton函数的参数需求
                import inspect
                sig = inspect.signature(triton_func)
                param_names = list(sig.parameters.keys())
                
                logger.info(f"Triton函数参数: {param_names}")
                logger.info(f"测试输入数量: {len(test_inputs)}")
                
                # 如果Triton函数需要更多参数（如weight, bias），尝试从PyTorch模型中提取
                if len(param_names) > len(test_inputs):
                    logger.info("Triton函数需要额外参数，尝试从PyTorch模型中提取...")
                    
                    # 尝试从pytorch_forward的闭包中获取模型
                    if hasattr(pytorch_forward, '__closure__') and pytorch_forward.__closure__:
                        for cell in pytorch_forward.__closure__:
                            if hasattr(cell.cell_contents, 'named_parameters'):
                                model = cell.cell_contents
                                logger.info("找到PyTorch模型，提取参数...")
                                
                                # 提取权重和偏置
                                model_params = dict(model.named_parameters())
                                
                                # 根据参数名称匹配
                                for param_name in param_names[len(test_inputs):]:
                                    if 'weight' in param_name.lower():
                                        if 'conv.weight' in model_params:
                                            additional_args.append(model_params['conv.weight'])
                                            logger.info(f"添加权重参数: {model_params['conv.weight'].shape}")
                                        elif 'weight' in model_params:
                                            additional_args.append(model_params['weight'])
                                            logger.info(f"添加权重参数: {model_params['weight'].shape}")
                                    elif 'bias' in param_name.lower():
                                        if 'bias' in model_params:
                                            additional_args.append(model_params['bias'])
                                            logger.info(f"添加偏置参数: {model_params['bias'].shape}")
                                        elif 'conv.bias' in model_params:
                                            additional_args.append(model_params['conv.bias'])
                                            logger.info(f"添加偏置参数: {model_params['conv.bias'].shape}")
                                break
                
                # 调用Triton函数
                if additional_args:
                    triton_result = triton_func(*test_inputs, *additional_args)
                    logger.info("成功调用Triton函数（包含额外参数）")
                else:
                    triton_result = triton_func(*test_inputs)
                    logger.info("调用Triton函数（仅使用测试输入）")
                
                if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                    max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
                    correctness = max_diff < 1e-4
                    
                    # 记录详细的正确性检查信息
                    logger.info(f"正确性检查详情:")
                    logger.info(f"  PyTorch输出形状: {pytorch_result.shape}")
                    logger.info(f"  Triton输出形状: {triton_result.shape}")
                    logger.info(f"  最大差异: {max_diff:.2e}")
                    logger.info(f"  正确性阈值: 1e-4")
                    logger.info(f"  正确性检查: {'通过' if correctness else '失败'}")
                    
                    if not correctness:
                        # 记录更多调试信息
                        logger.warning(f"正确性检查失败详情:")
                        logger.warning(f"  PyTorch输出统计: min={pytorch_result.min().item():.6f}, max={pytorch_result.max().item():.6f}, mean={pytorch_result.mean().item():.6f}")
                        logger.warning(f"  Triton输出统计: min={triton_result.min().item():.6f}, max={triton_result.max().item():.6f}, mean={triton_result.mean().item():.6f}")
                        
                        # 检查形状是否匹配
                        if pytorch_result.shape != triton_result.shape:
                            logger.error(f"输出形状不匹配: PyTorch {pytorch_result.shape} vs Triton {triton_result.shape}")
                else:
                    correctness = False
                    max_diff = float('inf')
                    logger.error(f"输出类型不匹配: PyTorch {type(pytorch_result)} vs Triton {type(triton_result)}")
                
            except Exception as e:
                logger.error(f"正确性检查执行失败: {e}")
                import traceback
                logger.error(f"详细错误信息:\n{traceback.format_exc()}")
                return {
                    "success": False,
                    "error": f"正确性检查失败: {e}",
                    "correctness": False,
                    "speedup": 0.0,
                    "pytorch_time": pytorch_time,
                    "triton_time": 0.0,
                    # "detailed_error": traceback.format_exc()
                }
            
            # 性能测试
            if correctness:
                try:
                    logger.info("正确性检查通过，开始性能测试...")
                    
                    # 为性能测试创建包装函数
                    if additional_args:
                        logger.info(f"性能测试使用额外参数数量: {len(additional_args)}")
                        
                        def triton_wrapper(*inputs):
                            return triton_func(*inputs, *additional_args)
                        
                        perf_result = self.benchmark.benchmark_general(
                            triton_wrapper, pytorch_forward, test_inputs
                        )
                    else:
                        perf_result = self.benchmark.benchmark_general(
                            triton_func, pytorch_forward, test_inputs
                        )
                    
                    if perf_result.get("success", False):
                        logger.info(f"性能测试完成: 加速比 {perf_result.get('speedup', 0.0):.2f}x")
                        return {
                            "success": True,
                            "correctness": True,
                            "speedup": perf_result.get("speedup", 0.0),
                            "pytorch_time": perf_result.get("pytorch_time_ms", 0.0),
                            "triton_time": perf_result.get("triton_time_ms", 0.0),
                            "max_diff": max_diff
                        }
                    else:
                        error_msg = f"性能测试失败: {perf_result.get('error', '未知错误')}"
                        logger.error(error_msg)
                        return {
                            "success": True,
                            "correctness": True,
                            "speedup": 0.0,
                            "error": error_msg,
                            "max_diff": max_diff
                        }
                        
                except Exception as e:
                    error_msg = f"性能测试失败: {e}"
                    logger.error(error_msg)
                    import traceback
                    logger.error(f"性能测试详细错误:\n{traceback.format_exc()}")
                    return {
                        "success": True,
                        "correctness": True,
                        "speedup": 0.0,
                        "error": error_msg,
                        "max_diff": max_diff
                        # "detailed_error": traceback.format_exc()
                    }
            else:
                error_msg = f"正确性检查未通过，最大差异: {max_diff:.2e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "correctness": False,
                    "speedup": 0.0,
                    "max_diff": max_diff,
                    "error": error_msg,
                    "pytorch_time": pytorch_time,
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

def create_performance_tools(config: Dict[str, Any]) -> list:
    """创建性能测试工具列表"""
    performance_config = config.get("performance", {})
    benchmark = TritonPerformanceBenchmark(
        device=performance_config.get("device", "cuda"),
        warmup_runs=performance_config.get("warmup_runs", 10),
        benchmark_runs=performance_config.get("benchmark_runs", 100)
    )
    
    return [PerformanceBenchmarkTool(benchmark)]