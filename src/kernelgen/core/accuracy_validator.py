"""
精度验证器 - 负责验证Triton kernel的数值精度和正确性
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import importlib.util

logger = logging.getLogger(__name__)


class AccuracyValidator:
    """
    精度验证器
    
    负责验证生成的Triton kernel的数值精度，
    包括与PyTorch参考实现的对比、数值稳定性测试等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化精度验证器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 精度验证配置
        self.accuracy_config = config.get("accuracy_validation", {})
        self.rtol = self.accuracy_config.get("rtol", 1e-5)  # 相对误差容忍度
        self.atol = self.accuracy_config.get("atol", 1e-8)  # 绝对误差容忍度
        self.test_cases = self.accuracy_config.get("test_cases", 10)  # 测试用例数量
        
        logger.info(f"AccuracyValidator初始化完成，设备: {self.device}")
    
    async def validate_accuracy(
        self,
        kernel_code: str,
        op_name: str,
        pytorch_code: str,
        input_shapes: List[tuple],
        dtype: str = "float32"
    ) -> Dict[str, Any]:
        """
        验证kernel精度
        
        Args:
            kernel_code: Triton kernel代码
            op_name: 算子名称
            pytorch_code: PyTorch参考实现代码
            input_shapes: 输入张量形状
            dtype: 数据类型
            
        Returns:
            精度验证结果字典
        """
        logger.info(f"开始精度验证: {op_name}")
        
        try:
            # Step 1: 编译Triton kernel
            triton_func = await self._compile_triton_kernel(kernel_code)
            if not triton_func:
                return {
                    "success": False,
                    "error": "Triton kernel编译失败"
                }
            
            # Step 2: 编译PyTorch参考实现
            pytorch_func = await self._compile_pytorch_reference(pytorch_code)
            if not pytorch_func:
                return {
                    "success": False,
                    "error": "PyTorch参考实现编译失败"
                }
            
            # Step 3: 生成测试用例
            test_cases = await self._generate_test_cases(input_shapes, dtype)
            
            # Step 4: 执行精度对比测试
            accuracy_results = []
            for i, test_inputs in enumerate(test_cases):
                result = await self._compare_outputs(
                    triton_func, pytorch_func, test_inputs, f"test_case_{i}"
                )
                accuracy_results.append(result)
            
            # Step 5: 数值稳定性测试
            stability_result = await self._test_numerical_stability(
                triton_func, input_shapes, dtype
            )
            
            # Step 6: 边界条件测试
            boundary_result = await self._test_boundary_conditions(
                triton_func, pytorch_func, input_shapes, dtype
            )
            
            # Step 7: 汇总结果
            overall_result = await self._summarize_results(
                accuracy_results, stability_result, boundary_result
            )
            
            logger.info(f"精度验证完成: {op_name}, 通过率: {overall_result['pass_rate']:.1f}%")
            
            return overall_result
            
        except Exception as e:
            logger.error(f"精度验证失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _compile_triton_kernel(self, kernel_code: str) -> Optional[Any]:
        """编译Triton kernel"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name
            
            try:
                spec = importlib.util.spec_from_file_location("triton_kernel", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找wrapper函数
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (callable(attr) and not attr_name.startswith('_') and 
                        not hasattr(attr, '__triton_jit__')):
                        return attr
                
                return None
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            logger.error(f"Triton kernel编译失败: {str(e)}")
            return None
    
    async def _compile_pytorch_reference(self, pytorch_code: str) -> Optional[Any]:
        """编译PyTorch参考实现"""
        try:
            # 提取Model类和forward方法
            namespace = {}
            exec(pytorch_code, namespace)
            
            if 'Model' in namespace:
                model_class = namespace['Model']
                model = model_class()
                model.eval()
                
                if torch.cuda.is_available():
                    model = model.cuda()
                
                return model.forward
            
            return None
            
        except Exception as e:
            logger.error(f"PyTorch参考实现编译失败: {str(e)}")
            return None
    
    async def _generate_test_cases(
        self,
        input_shapes: List[tuple],
        dtype: str
    ) -> List[List[torch.Tensor]]:
        """生成测试用例"""
        test_cases = []
        torch_dtype = getattr(torch, dtype)
        
        for _ in range(self.test_cases):
            test_inputs = []
            for shape in input_shapes:
                # 生成不同类型的测试数据
                case_type = np.random.choice(['normal', 'uniform', 'extreme'])
                
                if case_type == 'normal':
                    # 正态分布数据
                    tensor = torch.randn(shape, dtype=torch_dtype, device=self.device)
                elif case_type == 'uniform':
                    # 均匀分布数据
                    tensor = torch.rand(shape, dtype=torch_dtype, device=self.device) * 2 - 1
                else:
                    # 极值数据
                    tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
                    fill_value = np.random.choice([-1e6, -1e3, -1, 0, 1, 1e3, 1e6])
                    tensor.fill_(fill_value)
                
                test_inputs.append(tensor)
            
            test_cases.append(test_inputs)
        
        return test_cases
    
    async def _compare_outputs(
        self,
        triton_func: Any,
        pytorch_func: Any,
        test_inputs: List[torch.Tensor],
        case_name: str
    ) -> Dict[str, Any]:
        """比较Triton和PyTorch的输出"""
        try:
            # 执行PyTorch参考实现
            with torch.no_grad():
                pytorch_output = pytorch_func(*test_inputs)
            
            # 执行Triton实现
            if triton_func:
                triton_output = triton_func(*test_inputs)
            else:
                return {
                    "case_name": case_name,
                    "passed": False,
                    "error": "Triton函数不可用"
                }
            
            # 确保输出格式一致
            if not isinstance(pytorch_output, torch.Tensor):
                pytorch_output = torch.tensor(pytorch_output, device=self.device)
            if not isinstance(triton_output, torch.Tensor):
                triton_output = torch.tensor(triton_output, device=self.device)
            
            # 计算误差
            abs_error = torch.abs(triton_output - pytorch_output)
            rel_error = abs_error / (torch.abs(pytorch_output) + 1e-8)
            
            max_abs_error = torch.max(abs_error).item()
            max_rel_error = torch.max(rel_error).item()
            mean_abs_error = torch.mean(abs_error).item()
            mean_rel_error = torch.mean(rel_error).item()
            
            # 检查是否通过精度测试
            passed = torch.allclose(
                triton_output, pytorch_output, 
                rtol=self.rtol, atol=self.atol
            )
            
            return {
                "case_name": case_name,
                "passed": passed,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
                "mean_abs_error": mean_abs_error,
                "mean_rel_error": mean_rel_error,
                "output_shape": list(triton_output.shape)
            }
            
        except Exception as e:
            logger.error(f"输出比较失败 {case_name}: {str(e)}")
            return {
                "case_name": case_name,
                "passed": False,
                "error": str(e)
            }
    
    async def _test_numerical_stability(
        self,
        triton_func: Any,
        input_shapes: List[tuple],
        dtype: str
    ) -> Dict[str, Any]:
        """测试数值稳定性"""
        try:
            torch_dtype = getattr(torch, dtype)
            stability_results = []
            
            # 测试不同数值范围的稳定性
            test_ranges = [
                ("small", 1e-6, 1e-3),
                ("normal", -10, 10),
                ("large", 1e3, 1e6)
            ]
            
            for range_name, min_val, max_val in test_ranges:
                test_inputs = []
                for shape in input_shapes:
                    tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
                    tensor.uniform_(min_val, max_val)
                    test_inputs.append(tensor)
                
                try:
                    output = triton_func(*test_inputs)
                    
                    # 检查输出是否包含NaN或Inf
                    has_nan = torch.isnan(output).any().item()
                    has_inf = torch.isinf(output).any().item()
                    
                    stability_results.append({
                        "range": range_name,
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "stable": not (has_nan or has_inf)
                    })
                    
                except Exception as e:
                    stability_results.append({
                        "range": range_name,
                        "has_nan": True,
                        "has_inf": True,
                        "stable": False,
                        "error": str(e)
                    })
            
            # 计算稳定性得分
            stable_count = sum(1 for r in stability_results if r["stable"])
            stability_score = stable_count / len(stability_results)
            
            return {
                "stability_score": stability_score,
                "details": stability_results
            }
            
        except Exception as e:
            logger.error(f"数值稳定性测试失败: {str(e)}")
            return {
                "stability_score": 0.0,
                "error": str(e)
            }
    
    async def _test_boundary_conditions(
        self,
        triton_func: Any,
        pytorch_func: Any,
        input_shapes: List[tuple],
        dtype: str
    ) -> Dict[str, Any]:
        """测试边界条件"""
        try:
            torch_dtype = getattr(torch, dtype)
            boundary_results = []
            
            # 测试边界值
            boundary_values = [0.0, 1.0, -1.0]
            if dtype == "float32":
                boundary_values.extend([float('inf'), float('-inf')])
            
            for value in boundary_values:
                try:
                    test_inputs = []
                    for shape in input_shapes:
                        tensor = torch.full(shape, value, dtype=torch_dtype, device=self.device)
                        test_inputs.append(tensor)
                    
                    # 执行两个实现
                    with torch.no_grad():
                        pytorch_output = pytorch_func(*test_inputs)
                    triton_output = triton_func(*test_inputs)
                    
                    # 比较结果
                    if torch.isfinite(pytorch_output).all() and torch.isfinite(triton_output).all():
                        passed = torch.allclose(
                            triton_output, pytorch_output,
                            rtol=self.rtol, atol=self.atol
                        )
                    else:
                        # 对于非有限值，检查是否行为一致
                        passed = (torch.isnan(pytorch_output) == torch.isnan(triton_output)).all()
                    
                    boundary_results.append({
                        "value": value,
                        "passed": passed
                    })
                    
                except Exception as e:
                    boundary_results.append({
                        "value": value,
                        "passed": False,
                        "error": str(e)
                    })
            
            # 计算边界测试通过率
            passed_count = sum(1 for r in boundary_results if r["passed"])
            pass_rate = passed_count / len(boundary_results) if boundary_results else 0.0
            
            return {
                "boundary_pass_rate": pass_rate,
                "details": boundary_results
            }
            
        except Exception as e:
            logger.error(f"边界条件测试失败: {str(e)}")
            return {
                "boundary_pass_rate": 0.0,
                "error": str(e)
            }
    
    async def _summarize_results(
        self,
        accuracy_results: List[Dict[str, Any]],
        stability_result: Dict[str, Any],
        boundary_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """汇总验证结果"""
        try:
            # 计算精度测试通过率
            passed_count = sum(1 for r in accuracy_results if r.get("passed", False))
            accuracy_pass_rate = passed_count / len(accuracy_results) if accuracy_results else 0.0
            
            # 计算平均误差
            valid_results = [r for r in accuracy_results if "max_abs_error" in r]
            avg_max_abs_error = np.mean([r["max_abs_error"] for r in valid_results]) if valid_results else 0.0
            avg_max_rel_error = np.mean([r["max_rel_error"] for r in valid_results]) if valid_results else 0.0
            
            # 综合评分
            stability_score = stability_result.get("stability_score", 0.0)
            boundary_score = boundary_result.get("boundary_pass_rate", 0.0)
            
            overall_score = (accuracy_pass_rate * 0.5 + stability_score * 0.3 + boundary_score * 0.2)
            
            # 确定验证是否通过
            validation_passed = (
                accuracy_pass_rate >= 0.8 and
                stability_score >= 0.8 and
                boundary_score >= 0.6
            )
            
            return {
                "success": True,
                "validation_passed": validation_passed,
                "overall_score": overall_score,
                "pass_rate": accuracy_pass_rate * 100,
                "accuracy_results": {
                    "pass_rate": accuracy_pass_rate,
                    "avg_max_abs_error": avg_max_abs_error,
                    "avg_max_rel_error": avg_max_rel_error,
                    "details": accuracy_results
                },
                "stability_results": stability_result,
                "boundary_results": boundary_result,
                "summary": {
                    "total_tests": len(accuracy_results),
                    "passed_tests": passed_count,
                    "rtol": self.rtol,
                    "atol": self.atol
                }
            }
            
        except Exception as e:
            logger.error(f"结果汇总失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }