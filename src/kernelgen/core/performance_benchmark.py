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
        编译并加载Triton kernel - 改进版本
        
        优先选择wrapper函数而不是@triton.jit函数，确保可以直接调用。
        选择逻辑：
        1. 优先选择包含'triton_', 'forward', 'wrapper'的函数
        2. 确保函数有参数（不是无参数函数）
        3. 排除@triton.jit装饰的函数和test_开头的函数
        
        Args:
            kernel_code: 完整的Triton实现代码（包含kernel、wrapper、test函数）
            kernel_name: kernel名称（用于临时文件命名）
            
        Returns:
            编译后的wrapper函数（可直接调用），失败返回None
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
                
                # 简化的函数选择逻辑 - 结合两个系统的优点
                wrapper_patterns = ['triton_', 'forward', 'wrapper']
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if not callable(attr) or attr_name.startswith('_') or attr_name.startswith('test_'):
                        continue
                    
                    all_functions.append(attr_name)
                    
                    if hasattr(attr, '__triton_jit__'):
                        kernel_func = attr
                        logger.debug(f"找到@triton.jit函数: {attr_name}")
                    else:
                        # 简化的wrapper函数检查 - 学习原始系统的简洁性
                        try:
                            import inspect
                            sig = inspect.signature(attr)
                            if len(sig.parameters) > 0:  # wrapper函数应该有参数
                                # 优先选择匹配模式的函数
                                is_priority = any(pattern in attr_name.lower() for pattern in wrapper_patterns)
                                if is_priority:
                                    wrapper_func = attr
                                    logger.debug(f"找到优先wrapper函数: {attr_name}")
                                    break  # 找到优先函数就停止搜索
                                elif wrapper_func is None:
                                    wrapper_func = attr
                                    logger.debug(f"候选wrapper函数: {attr_name}")
                        except Exception as e:
                            logger.debug(f"检查函数签名失败 {attr_name}: {e}")
                
                logger.info(f"模块中的所有函数: {all_functions}")
                
                # 优先返回wrapper函数，其次是kernel函数
                selected_func = wrapper_func or kernel_func
                if selected_func:
                    func_type = "wrapper" if selected_func == wrapper_func else "kernel"
                    logger.info(f"选择{func_type}函数: {selected_func.__name__}")
                    return selected_func
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