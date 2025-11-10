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
            
            # 运行PyTorch基准
            try:
                # init_inputs是用于模型初始化的，不是用于forward调用的
                # pytorch_forward已经是初始化好的模型的forward方法
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
            try:
                triton_result = triton_func(*test_inputs)
                
                if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                    max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
                    correctness = max_diff < 1e-4
                else:
                    correctness = False
                    max_diff = float('inf')
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"正确性检查失败: {e}",
                    "correctness": False,
                    "speedup": 0.0,
                    "pytorch_time": pytorch_time,
                    "triton_time": 0.0
                }
            
            # 性能测试
            if correctness:
                try:
                    perf_result = self.benchmark.benchmark_general(
                        triton_func, pytorch_forward, test_inputs
                    )
                    
                    if perf_result.get("success", False):
                        return {
                            "success": True,
                            "correctness": True,
                            "speedup": perf_result.get("speedup", 0.0),
                            "pytorch_time": perf_result.get("pytorch_time_ms", 0.0),
                            "triton_time": perf_result.get("triton_time_ms", 0.0),
                            "max_diff": max_diff
                        }
                    else:
                        return {
                            "success": True,
                            "correctness": True,
                            "speedup": 0.0,
                            "error": f"性能测试失败: {perf_result.get('error', '未知错误')}",
                            "max_diff": max_diff
                        }
                        
                except Exception as e:
                    return {
                        "success": True,
                        "correctness": True,
                        "speedup": 0.0,
                        "error": f"性能测试失败: {e}",
                        "max_diff": max_diff
                    }
            else:
                return {
                    "success": False,
                    "correctness": False,
                    "speedup": 0.0,
                    "max_diff": max_diff,
                    "error": "正确性检查未通过",
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