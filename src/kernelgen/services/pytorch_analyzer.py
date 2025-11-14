"""
PyTorch模型分析服务
提取PyTorch模型的真实参数、输出形状等信息
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List, Tuple, Optional
import inspect
import re

logger = logging.getLogger(__name__)

class PyTorchModelAnalyzer:
    """PyTorch模型分析器，提取模型的真实参数和行为"""
    
    def __init__(self):
        self.layer_analyzers = {
            nn.Conv2d: self._analyze_conv2d,
            nn.Linear: self._analyze_linear,
            nn.BatchNorm2d: self._analyze_batchnorm2d,
            nn.ReLU: self._analyze_relu,
            nn.MaxPool2d: self._analyze_maxpool2d,
            nn.AdaptiveAvgPool2d: self._analyze_adaptive_avgpool2d,
        }
    
    def analyze_model_execution(self, pytorch_forward, test_inputs, init_inputs, 
                              pytorch_code: str) -> Dict[str, Any]:
        """
        分析PyTorch模型的完整执行信息
        
        Args:
            pytorch_forward: PyTorch前向函数
            test_inputs: 测试输入
            init_inputs: 初始化参数
            pytorch_code: PyTorch源代码
            
        Returns:
            完整的模型分析结果
        """
        try:
            # 1. 执行模型获取真实输出
            with torch.no_grad():
                if torch.cuda.is_available():
                    test_inputs_cuda = [inp.cuda() if isinstance(inp, torch.Tensor) else inp 
                                      for inp in test_inputs]
                    pytorch_output = pytorch_forward(*test_inputs_cuda)
                else:
                    pytorch_output = pytorch_forward(*test_inputs)
            
            # 2. 创建模型实例进行分析
            model_instance = self._create_model_instance(pytorch_code, init_inputs)
            
            # 3. 分析模型结构和参数
            model_analysis = self._analyze_model_structure(model_instance)
            
            # 4. 分析forward方法的操作序列
            forward_analysis = self._analyze_forward_method(pytorch_code, model_instance)
            
            # 5. 提取层参数信息
            layer_params = self._extract_layer_parameters(model_instance)
            
            # 6. 计算真实的输出形状和数值范围
            output_analysis = self._analyze_output_characteristics(pytorch_output)
            
            return {
                "input_shapes": [inp.shape if isinstance(inp, torch.Tensor) else str(inp) 
                               for inp in test_inputs],
                "output_shape": pytorch_output.shape if isinstance(pytorch_output, torch.Tensor) else str(pytorch_output),
                "output_analysis": output_analysis,
                "model_structure": model_analysis,
                "forward_operations": forward_analysis,
                "layer_parameters": layer_params,
                "pytorch_defaults": self._extract_pytorch_defaults(model_instance),
                "execution_context": {
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "dtype": str(pytorch_output.dtype) if isinstance(pytorch_output, torch.Tensor) else "unknown"
                }
            }
            
        except Exception as e:
            logger.error(f"PyTorch模型分析失败: {e}")
            return {
                "error": str(e),
                "input_shapes": [inp.shape if isinstance(inp, torch.Tensor) else str(inp) 
                               for inp in test_inputs],
                "output_shape": "unknown",
                "model_structure": {},
                "forward_operations": [],
                "layer_parameters": {},
                "pytorch_defaults": {}
            }
    
    def _create_model_instance(self, pytorch_code: str, init_inputs) -> nn.Module:
        """创建PyTorch模型实例"""
        # 执行PyTorch代码
        namespace = {}
        exec(pytorch_code, namespace)
        
        # 创建模型实例
        Model = namespace['Model']
        model = Model(*init_inputs)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def _analyze_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型结构"""
        structure = {
            "layers": [],
            "parameters": {},
            "total_params": sum(p.numel() for p in model.parameters())
        }
        
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": {}
                }
                
                # 分析特定层类型
                if type(module) in self.layer_analyzers:
                    layer_info.update(self.layer_analyzers[type(module)](module))
                
                structure["layers"].append(layer_info)
        
        # 提取所有参数
        for name, param in model.named_parameters():
            structure["parameters"][name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad
            }
        
        return structure
    
    def _analyze_conv2d(self, conv: nn.Conv2d) -> Dict[str, Any]:
        """分析Conv2d层的详细参数"""
        return {
            "conv2d_params": {
                "in_channels": conv.in_channels,
                "out_channels": conv.out_channels,
                "kernel_size": conv.kernel_size,
                "stride": conv.stride,
                "padding": conv.padding,  # 关键！真实的padding值
                "dilation": conv.dilation,
                "groups": conv.groups,
                "bias": conv.bias is not None,
                "padding_mode": conv.padding_mode
            }
        }
    
    def _analyze_linear(self, linear: nn.Linear) -> Dict[str, Any]:
        """分析Linear层参数"""
        return {
            "linear_params": {
                "in_features": linear.in_features,
                "out_features": linear.out_features,
                "bias": linear.bias is not None
            }
        }
    
    def _analyze_batchnorm2d(self, bn: nn.BatchNorm2d) -> Dict[str, Any]:
        """分析BatchNorm2d层参数"""
        return {
            "batchnorm2d_params": {
                "num_features": bn.num_features,
                "eps": bn.eps,
                "momentum": bn.momentum,
                "affine": bn.affine,
                "track_running_stats": bn.track_running_stats
            }
        }
    
    def _analyze_relu(self, relu: nn.ReLU) -> Dict[str, Any]:
        """分析ReLU层参数"""
        return {
            "relu_params": {
                "inplace": relu.inplace
            }
        }
    
    def _analyze_maxpool2d(self, pool: nn.MaxPool2d) -> Dict[str, Any]:
        """分析MaxPool2d层参数"""
        return {
            "maxpool2d_params": {
                "kernel_size": pool.kernel_size,
                "stride": pool.stride,
                "padding": pool.padding,
                "dilation": pool.dilation,
                "return_indices": pool.return_indices,
                "ceil_mode": pool.ceil_mode
            }
        }
    
    def _analyze_adaptive_avgpool2d(self, pool: nn.AdaptiveAvgPool2d) -> Dict[str, Any]:
        """分析AdaptiveAvgPool2d层参数"""
        return {
            "adaptive_avgpool2d_params": {
                "output_size": pool.output_size
            }
        }
    
    def _analyze_forward_method(self, pytorch_code: str, model: nn.Module) -> List[Dict[str, Any]]:
        """分析forward方法的操作序列"""
        operations = []
        
        # 提取forward方法的代码
        forward_code = self._extract_forward_method(pytorch_code)
        
        if forward_code:
            # 解析操作序列
            lines = forward_code.strip().split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    op_info = self._parse_operation_line(line, i)
                    if op_info:
                        operations.append(op_info)
        
        return operations
    
    def _extract_forward_method(self, pytorch_code: str) -> str:
        """提取forward方法的代码"""
        lines = pytorch_code.split('\n')
        forward_lines = []
        in_forward = False
        indent_level = 0
        
        for line in lines:
            if 'def forward(' in line:
                in_forward = True
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if in_forward:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= indent_level:
                    break
                forward_lines.append(line)
        
        return '\n'.join(forward_lines)
    
    def _parse_operation_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """解析单行操作"""
        # 匹配常见的操作模式
        patterns = [
            (r'x\s*=\s*self\.(\w+)\(([^)]*)\)', 'layer_call'),
            (r'x\s*=\s*torch\.(\w+)\(([^)]*)\)', 'torch_function'),
            (r'x\s*=\s*([^=]+)', 'assignment'),
        ]
        
        for pattern, op_type in patterns:
            match = re.search(pattern, line)
            if match:
                return {
                    "line_number": line_num,
                    "operation_type": op_type,
                    "raw_line": line,
                    "parsed_info": match.groups() if match.groups() else []
                }
        
        return None
    
    def _extract_layer_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """提取层的具体参数值"""
        layer_params = {}
        
        for name, module in model.named_modules():
            if name and hasattr(module, 'weight'):
                layer_params[name] = {
                    "weight_shape": list(module.weight.shape),
                    "weight_dtype": str(module.weight.dtype)
                }
                
                if hasattr(module, 'bias') and module.bias is not None:
                    layer_params[name]["bias_shape"] = list(module.bias.shape)
                    layer_params[name]["bias_dtype"] = str(module.bias.dtype)
                else:
                    layer_params[name]["bias_shape"] = None
        
        return layer_params
    
    def _extract_pytorch_defaults(self, model: nn.Module) -> Dict[str, Any]:
        """提取PyTorch层的默认参数"""
        defaults = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                defaults[f"{name}_conv2d"] = {
                    "default_padding": module.padding,
                    "default_stride": module.stride,
                    "default_dilation": module.dilation,
                    "default_groups": module.groups,
                    "has_bias": module.bias is not None
                }
            elif isinstance(module, nn.Linear):
                defaults[f"{name}_linear"] = {
                    "has_bias": module.bias is not None
                }
        
        return defaults
    
    def _analyze_output_characteristics(self, output: torch.Tensor) -> Dict[str, Any]:
        """分析输出张量的特征"""
        if not isinstance(output, torch.Tensor):
            return {"type": str(type(output)), "value": str(output)}
        
        return {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "device": str(output.device),
            "min_value": float(output.min().item()) if output.numel() > 0 else 0.0,
            "max_value": float(output.max().item()) if output.numel() > 0 else 0.0,
            "mean_value": float(output.mean().item()) if output.numel() > 0 else 0.0,
            "std_value": float(output.std().item()) if output.numel() > 0 else 0.0,
            "has_nan": bool(torch.isnan(output).any().item()) if output.numel() > 0 else False,
            "has_inf": bool(torch.isinf(output).any().item()) if output.numel() > 0 else False
        }
    
    def generate_analysis_context_string(self, analysis_result: Dict[str, Any]) -> str:
        """生成用于LLM的分析上下文字符串"""
        if "error" in analysis_result:
            return f"## PyTorch模型分析\n分析失败: {analysis_result['error']}"
        
        context = "## PyTorch模型真实执行分析\n\n"
        
        # 输入输出信息
        context += "### 输入输出信息\n"
        context += f"- 输入形状: {analysis_result['input_shapes']}\n"
        context += f"- 输出形状: {analysis_result['output_shape']}\n"
        context += f"- 输出数值范围: [{analysis_result['output_analysis'].get('min_value', 'N/A'):.6f}, {analysis_result['output_analysis'].get('max_value', 'N/A'):.6f}]\n\n"
        
        # 模型结构
        context += "### 模型层结构\n"
        for layer in analysis_result['model_structure']['layers']:
            context += f"- {layer['name']} ({layer['type']})\n"
            
            # 添加层的具体参数
            if 'conv2d_params' in layer:
                params = layer['conv2d_params']
                context += f"  * 卷积参数: in_channels={params['in_channels']}, out_channels={params['out_channels']}\n"
                context += f"  * 核大小: {params['kernel_size']}, 步长: {params['stride']}\n"
                context += f"  * **关键**: padding={params['padding']} (这是真实的padding值!)\n"
                context += f"  * 膨胀: {params['dilation']}, 偏置: {params['bias']}\n"
            
            elif 'linear_params' in layer:
                params = layer['linear_params']
                context += f"  * 线性层: {params['in_features']} → {params['out_features']}, 偏置: {params['bias']}\n"
        
        context += "\n"
        
        # Forward操作序列
        context += "### Forward操作序列\n"
        for i, op in enumerate(analysis_result['forward_operations']):
            context += f"{i+1}. {op['raw_line']}\n"
        context += "\n"
        
        # PyTorch默认参数
        context += "### PyTorch层默认参数 (关键信息)\n"
        for layer_name, defaults in analysis_result['pytorch_defaults'].items():
            context += f"- {layer_name}:\n"
            for param, value in defaults.items():
                context += f"  * {param}: {value}\n"
        
        context += "\n### 重要提醒\n"
        context += "- **必须使用上述真实的参数值，特别是padding、stride等**\n"
        context += "- **Triton输出形状必须与PyTorch输出形状完全一致**\n"
        context += f"- **目标输出形状: {analysis_result['output_shape']}**\n"
        
        return context