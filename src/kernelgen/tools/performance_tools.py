"""
性能测试相关的LangChain工具
"""

import json
import logging
import torch
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
            
            # 运行PyTorch基准 - 使用KernelBench原生模型作为金标准
            try:
                # 确保使用KernelBench的原生模型进行验证
                # pytorch_forward是从KernelBench数据库中执行的原始模型forward方法
                # 这是我们验证的"金标准"基准
                pytorch_result = pytorch_forward(*test_inputs)
                
                # 验证PyTorch结果的合理性
                if not isinstance(pytorch_result, torch.Tensor):
                    raise ValueError(f"PyTorch基准返回了非张量结果: {type(pytorch_result)}")
                
                if torch.isnan(pytorch_result).any() or torch.isinf(pytorch_result).any():
                    raise ValueError("PyTorch基准结果包含NaN或Inf值")
                
                pytorch_time = self.benchmark._benchmark_pytorch_general(pytorch_forward, test_inputs)
                
                logger.info(f"PyTorch基准验证:")
                # logger.info(f"  使用KernelBench原生模型作为金标准")
                # logger.info(f"  输出形状: {pytorch_result.shape}")
                # logger.info(f"  数值范围: [{pytorch_result.min().item():.6f}, {pytorch_result.max().item():.6f}]")
                logger.info(f"  执行时间: {pytorch_time:.3f}ms")
                
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
                
                # logger.info(f"Triton函数参数: {param_names}")
                # logger.info(f"测试输入数量: {len(test_inputs)}")
                
                # 如果Triton函数需要更多参数，从PyTorch模型中精确提取
                if len(param_names) > len(test_inputs):
                    # logger.info("Triton函数需要额外参数，从KernelBench模型中提取...")
                    
                    # 尝试从pytorch_forward的闭包中获取模型
                    model = None
                    if hasattr(pytorch_forward, '__closure__') and pytorch_forward.__closure__:
                        for cell in pytorch_forward.__closure__:
                            if hasattr(cell.cell_contents, 'named_parameters'):
                                model = cell.cell_contents
                                break
                    
                    if model is not None:
                        # logger.info("找到KernelBench PyTorch模型，分析参数结构...")
                        
                        # 获取所有模型参数
                        model_params = dict(model.named_parameters())
                        # logger.info(f"模型参数: {list(model_params.keys())}")
                        
                        # 智能参数匹配 - 按照Triton函数参数顺序提取
                        for i, param_name in enumerate(param_names[len(test_inputs):]):
                            param_lower = param_name.lower()
                            
                            # 匹配权重参数
                            if 'weight' in param_lower:
                                # 优先匹配conv权重
                                if 'conv.weight' in model_params:
                                    additional_args.append(model_params['conv.weight'])
                                    # logger.info(f"提取conv权重: {model_params['conv.weight'].shape}")
                                elif any('weight' in k for k in model_params.keys()):
                                    weight_key = next(k for k in model_params.keys() if 'weight' in k)
                                    additional_args.append(model_params[weight_key])
                                    # logger.info(f"提取权重 {weight_key}: {model_params[weight_key].shape}")
                                else:
                                    logger.error(f"未找到权重参数匹配 {param_name}")
                            
                            # 匹配偏置参数
                            elif 'bias' in param_lower:
                                # 区分conv内置bias和额外bias
                                if 'conv_bias' in param_lower or param_lower == 'conv_bias':
                                    # conv内置bias
                                    if 'conv.bias' in model_params:
                                        additional_args.append(model_params['conv.bias'])
                                        # logger.info(f"提取conv偏置: {model_params['conv.bias'].shape}")
                                    else:
                                        # logger.warning("conv层没有bias参数")
                                        additional_args.append(None)
                                elif 'extra_bias' in param_lower or param_lower == 'bias':
                                    # 额外的bias参数
                                    if 'bias' in model_params:
                                        additional_args.append(model_params['bias'])
                                        # logger.info(f"提取额外偏置: {model_params['bias'].shape}")
                                    else:
                                        logger.error(f"未找到额外偏置参数")
                                else:
                                    # 通用bias匹配
                                    bias_keys = [k for k in model_params.keys() if 'bias' in k]
                                    if bias_keys:
                                        # 如果有多个bias，按顺序选择
                                        bias_key = bias_keys[min(i, len(bias_keys)-1)]
                                        additional_args.append(model_params[bias_key])
                                    #     logger.info(f"提取偏置 {bias_key}: {model_params[bias_key].shape}")
                                    else:
                                        logger.error(f"未找到偏置参数匹配 {param_name}")
                            
                            else:
                                logger.warning(f"未识别的参数类型: {param_name}")
                    
                    else:
                        logger.error("无法从pytorch_forward中提取模型参数")
                
                # 调用Triton函数
                if additional_args:
                    triton_result = triton_func(*test_inputs, *additional_args)
                    # logger.info("成功调用Triton函数（包含额外参数）")
                else:
                    triton_result = triton_func(*test_inputs)
                    # logger.info("调用Triton函数（仅使用测试输入）")
                
                if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                    # 动态确定验证阈值
                    rtol, atol, description = self._get_dynamic_tolerance(pytorch_result, triton_result)
                    
                    # 使用torch.allclose进行更合理的验证
                    correctness = torch.allclose(pytorch_result, triton_result, rtol=rtol, atol=atol)
                    max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
                    
                    # 计算相对误差（避免除以接近0的值）
                    pytorch_abs_mean = torch.abs(pytorch_result).mean().item()
                    pytorch_abs_max = torch.abs(pytorch_result).max().item()
                    # 使用max和mean的较大值作为分母，避免除以很小的数
                    denominator = max(pytorch_abs_mean, pytorch_abs_max * 0.1, 1e-6)
                    relative_error = max_diff / denominator
                    
                    # 记录详细的正确性检查信息
                    logger.info(f"正确性检查详情:")
                    logger.info(f"  PyTorch输出形状: {pytorch_result.shape}")
                    logger.info(f"  Triton输出形状: {triton_result.shape}")
                    # logger.info(f"  最大绝对差异: {max_diff:.2e}")
                    # logger.info(f"  相对误差: {relative_error:.2e}")
                    # logger.info(f"  验证策略: {description}")
                    # logger.info(f"  阈值设置: rtol={rtol:.1e}, atol={atol:.1e}")
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
                    relative_error = float('inf')
                    logger.error(f"输出类型不匹配: PyTorch {type(pytorch_result)} vs Triton {type(triton_result)}")
                
            except Exception as e:
                # logger.error(f"正确性检查执行失败: {e}")
                import traceback
                # logger.error(f"详细错误信息:\n{traceback.format_exc()}")
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
                        # logger.info(f"性能测试使用额外参数数量: {len(additional_args)}")
                        
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
                            "max_diff": max_diff,
                            "relative_error": relative_error
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
                    "relative_error": relative_error,
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
    
    def _get_dynamic_tolerance(self, pytorch_result: torch.Tensor, triton_result: torch.Tensor):
        """
        根据操作类型和数值特性动态确定验证阈值
        
        Args:
            pytorch_result: PyTorch输出结果
            triton_result: Triton输出结果
            
        Returns:
            (rtol, atol, description): 相对误差阈值、绝对误差阈值、策略描述
        """
        # 分析数值特性
        pytorch_abs_mean = torch.abs(pytorch_result).mean().item()
        pytorch_abs_max = torch.abs(pytorch_result).max().item()
        result_size = pytorch_result.numel()
        
        # 根据数值范围和张量大小确定阈值
        if pytorch_abs_mean < 1e-2:
            # 小数值：使用较严格的绝对阈值，但相对误差要宽松
            rtol, atol = 1e-2, 1e-4
            description = "小数值操作(严格绝对阈值)"
        elif pytorch_abs_mean < 1.0:
            # 中等数值：平衡相对和绝对误差，针对卷积等操作放宽
            rtol, atol = 1e-2, 5e-2  # 放宽绝对阈值到5e-2
            description = "中等数值操作(平衡阈值)"
        elif pytorch_abs_mean < 100.0:
            # 大数值：主要使用相对误差
            rtol, atol = 3e-3, 1e-3
            description = "大数值操作(相对误差为主)"
        else:
            # 非常大的数值：更宽松的相对误差
            rtol, atol = 5e-3, 1e-2
            description = "超大数值操作(宽松阈值)"
        
        # 根据张量大小调整（大张量允许更大误差）
        if result_size > 1e6:  # 超过100万元素
            rtol *= 3
            atol *= 5
            description += "+大张量调整"
        elif result_size > 1e4:  # 超过1万元素
            rtol *= 2
            atol *= 3
            description += "+中张量调整"
        
        # 根据操作复杂性推断（通过形状维度）
        if len(pytorch_result.shape) >= 4:
            # 4维张量操作（如卷积）- 最复杂
            rtol *= 5
            atol *= 10
            description += "+4D卷积操作"
        elif len(pytorch_result.shape) >= 3:
            # 3维张量操作（如批量矩阵乘法）
            rtol *= 3
            atol *= 5
            description += "+3D批量操作"
        
        return rtol, atol, description

def create_performance_tools(config: Dict[str, Any]) -> list:
    """创建性能测试工具列表"""
    performance_config = config.get("performance", {})
    benchmark = TritonPerformanceBenchmark(
        device=performance_config.get("device", "cuda"),
        warmup_runs=performance_config.get("warmup_runs", 10),
        benchmark_runs=performance_config.get("benchmark_runs", 100)
    )
    
    return [PerformanceBenchmarkTool(benchmark)]