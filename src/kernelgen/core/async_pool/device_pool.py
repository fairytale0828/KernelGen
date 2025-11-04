"""
设备池管理 - 管理NVIDIA GPU设备的分配和释放
"""

import asyncio
import logging
import torch
from typing import List, Optional

logger = logging.getLogger(__name__)


class DevicePool:
    """
    NVIDIA GPU设备池管理器
    
    负责管理可用的GPU设备，支持并发任务的设备分配和释放。
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        初始化设备池
        
        Args:
            device_ids: GPU设备ID列表，如果为None则自动检测所有可用GPU
        """
        self.available_devices = asyncio.Queue()
        self.device_ids = device_ids or self._detect_available_devices()
        
        # 初始化设备队列
        for device_id in self.device_ids:
            self.available_devices.put_nowait(device_id)
        
        logger.info(f"设备池初始化完成，可用设备: {self.device_ids}")
    
    def _detect_available_devices(self) -> List[int]:
        """自动检测可用的GPU设备"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU模式")
            return [0]  # 使用0表示CPU
        
        device_count = torch.cuda.device_count()
        devices = list(range(device_count))
        logger.info(f"检测到 {device_count} 个CUDA设备")
        return devices
    
    async def acquire_device(self) -> int:
        """
        获取一个可用设备
        
        Returns:
            设备ID
        """
        device_id = await self.available_devices.get()
        logger.debug(f"分配设备: {device_id}")
        return device_id
    
    async def release_device(self, device_id: int):
        """
        释放设备
        
        Args:
            device_id: 要释放的设备ID
        """
        await self.available_devices.put(device_id)
        logger.debug(f"释放设备: {device_id}")
    
    def get_device_count(self) -> int:
        """获取设备总数"""
        return len(self.device_ids)
    
    def get_available_count(self) -> int:
        """获取当前可用设备数"""
        return self.available_devices.qsize()
    
    def is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        return torch.cuda.is_available() and len(self.device_ids) > 0 and self.device_ids[0] != 0