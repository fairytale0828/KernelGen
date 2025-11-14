"""
硬件信息服务
"""

import logging
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeviceInfo:
    """设备信息"""
    name: str
    compute_capability: str
    memory_size: str
    sm_count: int
    is_cuda: bool
    device_id: int = 0

class HardwareInfoService:
    """硬件信息服务"""
    
    def __init__(self):
        self._cached_device_info: Optional[DeviceInfo] = None
    
    def get_current_device_info(self) -> DeviceInfo:
        """获取当前设备信息"""
        if self._cached_device_info is not None:
            return self._cached_device_info
        
        try:
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_id)
                
                device_info = DeviceInfo(
                    name=props.name,
                    compute_capability=f"{props.major}.{props.minor}",
                    memory_size=f"{props.total_memory / 1024**3:.1f}GB",
                    sm_count=props.multi_processor_count,
                    is_cuda=True,
                    device_id=device_id
                )
            else:
                device_info = DeviceInfo(
                    name="CPU",
                    compute_capability="N/A",
                    memory_size="N/A",
                    sm_count=0,
                    is_cuda=False
                )
            
            self._cached_device_info = device_info
            logger.info(f"检测到设备: {device_info.name}")
            return device_info
            
        except Exception as e:
            logger.error(f"获取硬件信息失败: {e}")
            # 返回默认CPU信息
            return DeviceInfo(
                name="Unknown",
                compute_capability="N/A", 
                memory_size="N/A",
                sm_count=0,
                is_cuda=False
            )
    
    def get_hardware_context_string(self) -> str:
        """获取硬件上下文字符串，用于prompt"""
        device_info = self.get_current_device_info()
        
        context = f"""## GPU硬件信息
- 设备型号: {device_info.name}
- 计算能力: {device_info.compute_capability}
- 内存大小: {device_info.memory_size}
- SM数量: {device_info.sm_count}

## 硬件优化建议
- 内存合并访问: 确保连续内存访问模式，提高带宽利用率
- 共享内存利用: 充分利用片上高速缓存，减少全局内存访问
- 线程束效率: 避免分支分歧，保持32线程束内的同步执行
- 寄存器使用: 平衡寄存器使用和SM占用率，避免寄存器溢出"""
        
        return context
    
    def get_optimization_recommendations(self, operation_type: str) -> Dict[str, Any]:
        """根据硬件特性和操作类型获取优化建议"""
        device_info = self.get_current_device_info()
        
        recommendations = {
            "block_size_suggestions": [],
            "memory_optimizations": [],
            "compute_optimizations": []
        }
        
        if device_info.is_cuda:
            # 基于计算能力的建议
            if device_info.compute_capability >= "8.0":  # A100等
                recommendations["block_size_suggestions"] = [256, 512, 1024]
                recommendations["memory_optimizations"] = [
                    "利用A100的高带宽内存",
                    "使用tensor core进行混合精度计算"
                ]
            elif device_info.compute_capability >= "7.0":  # V100等
                recommendations["block_size_suggestions"] = [128, 256, 512]
                recommendations["memory_optimizations"] = [
                    "优化内存合并访问",
                    "使用共享内存缓存"
                ]
            
            # 基于操作类型的建议
            if operation_type in ["matmul", "conv2d"]:
                recommendations["compute_optimizations"] = [
                    "使用分块算法",
                    "优化数据重用"
                ]
        
        return recommendations