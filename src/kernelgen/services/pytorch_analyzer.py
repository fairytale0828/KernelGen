"""
简化的PyTorch模型分析器
直接利用HuggingFace KernelBench提供的标杆PyTorch模型信息
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PyTorchModelAnalyzer:
    """
    简化的PyTorch模型分析器
    直接从KernelBench提供的pytorch_forward函数中提取所需信息
    """
    
    def analyze_from_benchmark(self, pytorch_forward, test_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        从KernelBench的pytorch_forward函数中分析模型信息
        
        Args:
            pytorch_forward: KernelBench提供的PyTorch前向函数
            test_inputs: 测试输入张量
            
        Returns:
            模型分析结果
        """
        try:
            # 1. 获取真实输出形状
            target_shape = self._get_target_output_shape(pytorch_forward, test_inputs)
            
            # 2. 从pytorch_forward中提取模型
            model = self._extract_model_from_forward(pytorch_forward)
            
            # 3. 分析模型层参数
            layer_info = self._analyze_model_layers(model) if model else {}
            
            # 4. 生成分析摘要
            summary = self._generate_simple_summary(target_shape, layer_info, test_inputs)
            
            return {
                "success": True,
                "target_output_shape": target_shape,
                "layer_parameters": layer_info,
                "analysis_summary": summary,
                "input_shapes": [inp.shape for inp in test_inputs]
            }
            
        except Exception as e:
            logger.error(f"PyTorch模型分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_summary": "无法分析PyTorch模型，请手动检查层参数"
            }
    
    def _get_target_output_shape(self, pytorch_forward, test_inputs: List[torch.Tensor]):
        """获取真实的输出形状"""
        with torch.no_grad():
            output = pytorch_forward(*test_inputs)
            return output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]
    
    def _extract_model_from_forward(self, pytorch_forward) -> Optional[nn.Module]:
        """从pytorch_forward函数中提取模型"""
        try:
            # 检查闭包中是否有模型
            if hasattr(pytorch_forward, '__closure__') and pytorch_forward.__closure__:
                for cell in pytorch_forward.__closure__:
                    if hasattr(cell.cell_contents, 'named_modules'):
                        return cell.cell_contents
            return None
        except Exception as e:
            logger.warning(f"无法从pytorch_forward中提取模型: {e}")
            return None
    
    def _analyze_model_layers(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型层参数"""
        layer_info = {}
        
        for name, module in model.named_modules():
            if name == '':  # 跳过根模块
                continue
            
            module_type = type(module).__name__
            
            # 分析常见层类型
            if isinstance(module, nn.Conv2d):
                layer_info[name] = {
                    'type': 'Conv2d',
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'bias': module.bias is not None
                }
            elif isinstance(module, nn.Linear):
                layer_info[name] = {
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None
                }
            elif isinstance(module, nn.ReLU):
                layer_info[name] = {
                    'type': 'ReLU',
                    'inplace': module.inplace
                }
            elif isinstance(module, nn.BatchNorm2d):
                layer_info[name] = {
                    'type': 'BatchNorm2d',
                    'num_features': module.num_features,
                    'eps': module.eps,
                    'momentum': module.momentum
                }
            else:
                # 其他层类型
                layer_info[name] = {
                    'type': module_type,
                    'parameters': {pname: param.shape for pname, param in module.named_parameters()}
                }
        
        return layer_info
    
    def _generate_simple_summary(self, target_shape, layer_info: Dict[str, Any], 
                                test_inputs: List[torch.Tensor]) -> str:
        """生成简单的分析摘要"""
        summary = []
        summary.append("## PyTorch模型分析")
        
        # 输入输出形状
        input_shapes = [inp.shape for inp in test_inputs]
        summary.append(f"- 输入形状: {input_shapes}")
        summary.append(f"- 输出形状: {target_shape}")
        
        # 关键层信息
        if layer_info:
            summary.append("## 关键层参数")
            for name, info in layer_info.items():
                layer_type = info.get('type', 'Unknown')
                if layer_type == 'Conv2d':
                    summary.append(f"- {name} (Conv2d): {info['in_channels']}→{info['out_channels']}, "
                                 f"kernel={info['kernel_size']}, padding={info['padding']}, stride={info['stride']}")
                elif layer_type == 'Linear':
                    summary.append(f"- {name} (Linear): {info['in_features']}→{info['out_features']}, bias={info['bias']}")
                elif layer_type in ['ReLU', 'BatchNorm2d']:
                    summary.append(f"- {name} ({layer_type})")
        
        return "\n".join(summary)