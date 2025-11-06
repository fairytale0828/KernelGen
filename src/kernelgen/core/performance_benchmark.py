"""
真实的Triton性能基准测试模块
集成KernelBench的评估方法
"""
import torch
import time
import tempfile
import os
import importlib.util
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import subprocess
import sys

logger = logging.getLogger(__name__)

class TritonPerformanceBenchmark:
    """
    真实的Triton kernel性能基准测试器
    """
    
    def __init__(self, device: str = "cuda", warmup_runs: int = 10, benchmark_runs: int = 100):
        """
        初始化性能基准测试器
        
        Args:
            device: 测试设备
            warmup_runs: 预热运行次数
            benchmark_runs: 基准测试运行次数
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU模式")
            self.device = "cpu"
    
    def compile_and_load_kernel(self, kernel_code: str, kernel_name: str = "test_kernel") -> Optional[Any]:
        """
        编译并加载Triton kernel
        
        Args:
            kernel_code: Triton kernel代码
            kernel_name: kernel名称
            
        Returns:
            编译后的kernel函数，失败返回None
        """
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name
            
            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(kernel_name, temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找kernel函数和wrapper函数
                kernel_func = None
                wrapper_func = None
                all_functions = []
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if not callable(attr) or attr_name.startswith('_'):
                        continue
                    
                    all_functions.append(attr_name)
                    
                    if hasattr(attr, '__triton_jit__'):
                        kernel_func = attr
                        logger.debug(f"找到@triton.jit函数: {attr_name}")
                    elif not attr_name.startswith('test_'):  # 排除测试函数
                        # 检查函数签名，确保它可以接受参数
                        import inspect
                        try:
                            sig = inspect.signature(attr)
                            param_count = len(sig.parameters)
                            
                            # wrapper函数应该有参数
                            if param_count > 0:
                                # 优先选择包含操作名称的函数
                                if ('triton' in attr_name.lower() or 
                                    'relu' in attr_name.lower() or
                                    'wrapper' in attr_name.lower() or
                                    kernel_name.replace('_kernel', '') in attr_name.lower()):
                                    wrapper_func = attr
                                    logger.debug(f"找到wrapper函数: {attr_name} (参数数量: {param_count})")
                                elif wrapper_func is None:  # 如果还没找到wrapper函数，这个也可以作为候选
                                    wrapper_func = attr
                                    logger.debug(f"候选wrapper函数: {attr_name} (参数数量: {param_count})")
                        except Exception as e:
                            logger.debug(f"检查函数签名失败 {attr_name}: {e}")
                
                logger.info(f"模块中的所有函数: {all_functions}")
                
                # 优先返回wrapper函数，其次是kernel函数
                if wrapper_func:
                    logger.info(f"选择wrapper函数: {wrapper_func.__name__}")
                    return wrapper_func
                elif kernel_func:
                    logger.info(f"选择kernel函数: {kernel_func.__name__}")
                    return kernel_func
                else:
                    logger.error("未找到可调用的kernel函数")
                    return None
                    
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"编译kernel失败: {str(e)}")
            return None
    
    def validate_kernel_syntax(self, kernel_code: str) -> Tuple[bool, str]:
        """
        验证kernel语法
        
        Args:
            kernel_code: kernel代码
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 基本语法检查
            compile(kernel_code, "<string>", "exec")
            
            # 检查必要的导入和装饰器
            required_patterns = [
                "import torch",
                "@triton.jit" if "triton" in kernel_code else "def ",
                "def "
            ]
            
            for pattern in required_patterns:
                if pattern not in kernel_code:
                    return False, f"缺少必要的语法元素: {pattern}"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"语法错误: {str(e)}"
        except Exception as e:
            return False, f"验证失败: {str(e)}"
    
    def benchmark_matmul(self, triton_func: Any, shapes: List[Tuple[int, int, int]], 
                        dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        对MatMul kernel进行性能基准测试
        
        Args:
            triton_func: 编译后的Triton函数
            shapes: 测试形状列表 [(M, N, K), ...]
            dtype: 数据类型
            
        Returns:
            性能测试结果
        """
        results = {
            "shapes": shapes,
            "triton_times": [],
            "pytorch_times": [],
            "triton_gflops": [],
            "pytorch_gflops": [],
            "speedups": [],
            "success": True,
            "error": None
        }
        
        try:
            for M, N, K in shapes:
                logger.info(f"测试形状 [{M}, {N}, {K}]...")
                
                # 创建测试数据
                if self.device == "cuda":
                    a = torch.randn(M, K, dtype=dtype, device="cuda")
                    b = torch.randn(K, N, dtype=dtype, device="cuda")
                    c_torch = torch.zeros(M, N, dtype=dtype, device="cuda")
                    c_triton = torch.zeros(M, N, dtype=dtype, device="cuda")
                else:
                    a = torch.randn(M, K, dtype=dtype)
                    b = torch.randn(K, N, dtype=dtype)
                    c_torch = torch.zeros(M, N, dtype=dtype)
                    c_triton = torch.zeros(M, N, dtype=dtype)
                
                # PyTorch基准测试
                pytorch_time = self._benchmark_pytorch_matmul(a, b, c_torch)
                
                # Triton基准测试
                triton_time = self._benchmark_triton_matmul(triton_func, a, b, c_triton)
                
                if triton_time is None:
                    logger.error(f"Triton kernel测试失败，形状: [{M}, {N}, {K}]")
                    continue
                
                # 计算GFLOPS
                flops = 2 * M * N * K  # MatMul的FLOP数
                pytorch_gflops = flops / (pytorch_time * 1e-3) / 1e9  # 转换为GFLOPS
                triton_gflops = flops / (triton_time * 1e-3) / 1e9
                
                speedup = pytorch_time / triton_time if triton_time > 0 else 0
                
                results["triton_times"].append(triton_time)
                results["pytorch_times"].append(pytorch_time)
                results["triton_gflops"].append(triton_gflops)
                results["pytorch_gflops"].append(pytorch_gflops)
                results["speedups"].append(speedup)
                
                logger.info(f"  PyTorch: {pytorch_time:.3f}ms ({pytorch_gflops:.1f} GFLOPS)")
                logger.info(f"  Triton:  {triton_time:.3f}ms ({triton_gflops:.1f} GFLOPS)")
                logger.info(f"  加速比: {speedup:.2f}x")
                
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            logger.error(f"性能测试失败: {str(e)}")
        
        return results
    
    def _benchmark_pytorch_matmul(self, a: torch.Tensor, b: torch.Tensor, 
                                 c: torch.Tensor) -> float:
        """PyTorch MatMul性能测试"""
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # 预热
        for _ in range(self.warmup_runs):
            torch.mm(a, b, out=c)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # 基准测试
        start_time = time.perf_counter()
        for _ in range(self.benchmark_runs):
            torch.mm(a, b, out=c)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / self.benchmark_runs * 1000  # 转换为毫秒
        return avg_time
    
    def _benchmark_triton_matmul(self, triton_func: Any, a: torch.Tensor, 
                                b: torch.Tensor, c: torch.Tensor) -> Optional[float]:
        """Triton MatMul性能测试"""
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 预热
            for _ in range(self.warmup_runs):
                triton_func(a, b, c)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 基准测试
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                triton_func(a, b, c)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / self.benchmark_runs * 1000  # 转换为毫秒
            return avg_time
            
        except Exception as e:
            logger.error(f"Triton kernel执行失败: {str(e)}")
            return None
    
    def benchmark_general(self, triton_func: Any, pytorch_func: Any, test_inputs: List[torch.Tensor], 
                         dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        通用性能基准测试 - 适用于任何算子类型
        
        Args:
            triton_func: 编译后的Triton函数
            pytorch_func: PyTorch参考函数
            test_inputs: 测试输入列表
            dtype: 数据类型
            
        Returns:
            性能测试结果
        """
        results = {
            "triton_times": [],
            "pytorch_times": [],
            "speedups": [],
            "success": True,
            "error": None,
            "has_error": False
        }
        
        try:
            logger.info(f"开始通用性能测试...")
            
            # PyTorch基准测试
            pytorch_time = self._benchmark_pytorch_general(pytorch_func, test_inputs)
            
            # Triton基准测试
            triton_time = self._benchmark_triton_general(triton_func, test_inputs)
            
            if triton_time is None:
                results["error"] = "Triton kernel执行失败"
                results["success"] = False
                results["has_error"] = True
                return results
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            results["triton_times"].append(triton_time)
            results["pytorch_times"].append(pytorch_time)
            results["speedups"].append(speedup)
            results["speedup"] = speedup  # 添加单个speedup字段
            results["triton_time_ms"] = triton_time
            results["pytorch_time_ms"] = pytorch_time
            
            logger.info(f"  PyTorch: {pytorch_time:.3f}ms")
            logger.info(f"  Triton:  {triton_time:.3f}ms")
            logger.info(f"  加速比: {speedup:.2f}x")
                
        except Exception as e:
            results["success"] = False
            results["has_error"] = True
            results["error"] = str(e)
            logger.error(f"性能测试失败: {str(e)}")
        
        return results
    
    def _benchmark_pytorch_general(self, pytorch_func: Any, test_inputs: List[torch.Tensor]) -> float:
        """通用PyTorch性能测试"""
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # 预热
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                pytorch_func(*test_inputs)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # 基准测试
        start_time = time.perf_counter()
        for _ in range(self.benchmark_runs):
            with torch.no_grad():
                pytorch_func(*test_inputs)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / self.benchmark_runs * 1000
        return avg_time
    
    def _benchmark_triton_general(self, triton_func: Any, test_inputs: List[torch.Tensor]) -> Optional[float]:
        """通用Triton性能测试"""
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 预热
            for _ in range(self.warmup_runs):
                triton_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 基准测试
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                triton_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / self.benchmark_runs * 1000
            return avg_time
            
        except Exception as e:
            logger.error(f"Triton kernel执行失败: {str(e)}")
            return None