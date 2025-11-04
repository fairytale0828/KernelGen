"""
性能调优器 - 负责Triton kernel的性能优化和调优
"""

import logging
import time
import torch
import triton
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import importlib.util

logger = logging.getLogger(__name__)


class PerformanceTuner:
    """
    性能调优器
    
    负责对生成的Triton kernel进行性能调优，
    包括block size优化、内存访问优化、计算优化等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化性能调优器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 性能调优配置
        self.tuning_config = config.get("performance_tuning", {})
        self.warmup_runs = self.tuning_config.get("warmup_runs", 10)
        self.benchmark_runs = self.tuning_config.get("benchmark_runs", 100)
        self.block_sizes = self.tuning_config.get("block_sizes", [32, 64, 128, 256, 512, 1024])
        
        logger.info(f"PerformanceTuner初始化完成，设备: {self.device}")
    
    async def tune_kernel(
        self,
        kernel_code: str,
        op_name: str,
        input_shapes: List[tuple],
        dtype: str = "float32"
    ) -> Dict[str, Any]:
        """
        对kernel进行性能调优
        
        Args:
            kernel_code: Triton kernel代码
            op_name: 算子名称
            input_shapes: 输入张量形状
            dtype: 数据类型
            
        Returns:
            调优结果字典
        """
        logger.info(f"开始性能调优: {op_name}")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，跳过性能调优")
            return {
                "success": False,
                "error": "CUDA不可用",
                "optimized_code": kernel_code
            }
        
        try:
            # Step 1: 编译kernel
            kernel_func, wrapper_func = await self._compile_kernel(kernel_code)
            if not kernel_func:
                return {
                    "success": False,
                    "error": "kernel编译失败",
                    "optimized_code": kernel_code
                }
            
            # Step 2: Block size调优
            best_block_size, best_performance = await self._tune_block_size(
                kernel_func, wrapper_func, input_shapes, dtype
            )
            
            # Step 3: 生成优化后的代码
            optimized_code = await self._generate_optimized_code(
                kernel_code, best_block_size, best_performance
            )
            
            # Step 4: 验证优化效果
            optimization_gain = await self._validate_optimization(
                kernel_code, optimized_code, input_shapes, dtype
            )
            
            logger.info(f"性能调优完成: {op_name}, 性能提升: {optimization_gain:.2f}%")
            
            return {
                "success": True,
                "optimized_code": optimized_code,
                "best_block_size": best_block_size,
                "performance_metrics": best_performance,
                "optimization_gain": optimization_gain,
                "tuning_details": {
                    "tested_block_sizes": self.block_sizes,
                    "warmup_runs": self.warmup_runs,
                    "benchmark_runs": self.benchmark_runs
                }
            }
            
        except Exception as e:
            logger.error(f"性能调优失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "optimized_code": kernel_code
            }
    
    async def _compile_kernel(self, kernel_code: str) -> Tuple[Optional[Any], Optional[Any]]:
        """编译kernel并提取函数"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name
            
            try:
                # 导入模块
                spec = importlib.util.spec_from_file_location("temp_kernel", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找kernel函数和wrapper函数
                kernel_func = None
                wrapper_func = None
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, '__triton_jit__'):
                        kernel_func = attr
                    elif callable(attr) and not attr_name.startswith('_') and attr_name != kernel_func:
                        wrapper_func = attr
                
                return kernel_func, wrapper_func
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            logger.error(f"kernel编译失败: {str(e)}")
            return None, None
    
    async def _tune_block_size(
        self,
        kernel_func: Any,
        wrapper_func: Any,
        input_shapes: List[tuple],
        dtype: str
    ) -> Tuple[int, Dict[str, float]]:
        """调优block size"""
        best_block_size = 128
        best_performance = {"avg_time_ms": float('inf')}
        
        logger.info(f"开始block size调优，测试范围: {self.block_sizes}")
        
        for block_size in self.block_sizes:
            try:
                # 创建测试数据
                test_inputs = self._create_test_inputs(input_shapes, dtype)
                
                # 真实性能测试
                performance = await self._real_benchmark_kernel(
                    wrapper_func, test_inputs, block_size
                )
                
                logger.debug(f"Block size {block_size}: {performance['avg_time_ms']:.3f}ms")
                
                # 更新最佳配置
                if performance["avg_time_ms"] < best_performance["avg_time_ms"]:
                    best_block_size = block_size
                    best_performance = performance
                    
            except Exception as e:
                logger.warning(f"Block size {block_size} 测试失败: {str(e)}")
                continue
        
        logger.info(f"最佳block size: {best_block_size}, 性能: {best_performance['avg_time_ms']:.3f}ms")
        return best_block_size, best_performance
    
    def _create_test_inputs(self, input_shapes: List[tuple], dtype: str) -> List[torch.Tensor]:
        """创建测试输入数据"""
        test_inputs = []
        torch_dtype = getattr(torch, dtype)
        
        for shape in input_shapes:
            tensor = torch.randn(shape, dtype=torch_dtype, device=self.device)
            test_inputs.append(tensor)
        
        return test_inputs
    
    async def _real_benchmark_kernel(
        self,
        wrapper_func: Any,
        test_inputs: List[torch.Tensor],
        block_size: int
    ) -> Dict[str, float]:
        """对kernel进行真实性能基准测试"""
        try:
            if not torch.cuda.is_available() or not wrapper_func:
                return {"avg_time_ms": float('inf')}
            
            # 确保输入在正确的设备上
            test_inputs = [inp.cuda() if inp.device.type != 'cuda' else inp for inp in test_inputs]
            
            # 预热运行
            for _ in range(self.warmup_runs):
                try:
                    wrapper_func(*test_inputs)
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"预热运行失败: {str(e)}")
                    continue
            
            # 使用CUDA事件进行精确计时
            times = []
            successful_runs = 0
            
            for _ in range(self.benchmark_runs):
                try:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    torch.cuda.synchronize()
                    start_event.record()
                    
                    # 执行kernel
                    wrapper_func(*test_inputs)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    # 获取执行时间
                    elapsed_time = start_event.elapsed_time(end_event)
                    times.append(elapsed_time)
                    successful_runs += 1
                    
                except Exception as e:
                    logger.warning(f"基准测试运行失败: {str(e)}")
                    times.append(float('inf'))
            
            # 过滤有效时间
            valid_times = [t for t in times if t != float('inf') and t > 0]
            if not valid_times:
                return {
                    "avg_time_ms": float('inf'),
                    "error": "所有基准测试都失败",
                    "successful_runs": successful_runs,
                    "total_runs": self.benchmark_runs
                }
            
            # 计算真实统计指标
            avg_time = np.mean(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            std_time = np.std(valid_times)
            median_time = np.median(valid_times)
            
            # 计算真实吞吐量
            total_elements = sum(np.prod(inp.shape) for inp in test_inputs)
            throughput_elements_per_sec = total_elements / (avg_time / 1000)
            
            # 估算GFLOPS (假设每个元素1个操作)
            throughput_gflops = throughput_elements_per_sec / 1e9
            
            # 计算内存带宽利用率
            bytes_per_element = test_inputs[0].element_size()
            total_bytes = total_elements * bytes_per_element * 2  # 读+写
            memory_bandwidth_gb_s = (total_bytes / (avg_time / 1000)) / 1e9
            
            return {
                "avg_time_ms": float(avg_time),
                "min_time_ms": float(min_time),
                "max_time_ms": float(max_time),
                "median_time_ms": float(median_time),
                "std_time_ms": float(std_time),
                "throughput_gflops": float(throughput_gflops),
                "memory_bandwidth_gb_s": float(memory_bandwidth_gb_s),
                "successful_runs": successful_runs,
                "total_runs": self.benchmark_runs,
                "success_rate": successful_runs / self.benchmark_runs
            }
            
        except Exception as e:
            error_msg = f"真实性能测试异常: {str(e)}"
            logger.error(error_msg)
            return {"avg_time_ms": float('inf'), "error": error_msg}
    
    async def _generate_optimized_code(
        self,
        original_code: str,
        best_block_size: int,
        performance_metrics: Dict[str, float]
    ) -> str:
        """生成优化后的代码"""
        try:
            # 在代码中更新BLOCK_SIZE
            optimized_code = original_code
            
            # 添加性能优化注释
            optimization_comment = f"""
# 性能优化配置
# 最佳BLOCK_SIZE: {best_block_size}
# 平均执行时间: {performance_metrics.get('avg_time_ms', 0):.3f}ms
# 吞吐量: {performance_metrics.get('throughput_gflops', 0):.2f} GFLOPS
"""
            
            # 在import语句后添加优化注释
            lines = optimized_code.split('\n')
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import'):
                    insert_index = i + 1
            
            lines.insert(insert_index, optimization_comment)
            
            # 更新BLOCK_SIZE默认值
            for i, line in enumerate(lines):
                if 'BLOCK_SIZE' in line and '=' in line and 'constexpr' not in line:
                    # 更新默认BLOCK_SIZE
                    lines[i] = line.replace(
                        line.split('=')[1].strip(),
                        str(best_block_size)
                    )
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"生成优化代码失败: {str(e)}")
            return original_code
    
    async def _validate_optimization(
        self,
        original_code: str,
        optimized_code: str,
        input_shapes: List[tuple],
        dtype: str
    ) -> float:
        """验证优化效果"""
        try:
            # 编译原始代码
            orig_kernel, orig_wrapper = await self._compile_kernel(original_code)
            
            # 编译优化代码
            opt_kernel, opt_wrapper = await self._compile_kernel(optimized_code)
            
            if not (orig_wrapper and opt_wrapper):
                return 0.0
            
            # 创建测试数据
            test_inputs = self._create_test_inputs(input_shapes, dtype)
            
            # 测试原始性能
            orig_perf = await self._benchmark_kernel(orig_wrapper, test_inputs, 128)
            
            # 测试优化性能
            opt_perf = await self._benchmark_kernel(opt_wrapper, test_inputs, 128)
            
            # 计算性能提升
            if orig_perf["avg_time_ms"] > 0 and opt_perf["avg_time_ms"] > 0:
                improvement = ((orig_perf["avg_time_ms"] - opt_perf["avg_time_ms"]) / 
                              orig_perf["avg_time_ms"]) * 100
                return max(0.0, improvement)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"优化验证失败: {str(e)}")
            return 0.0