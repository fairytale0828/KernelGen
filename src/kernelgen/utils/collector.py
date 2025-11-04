"""
数据收集器 - 收集和管理执行数据
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 全局收集器实例
_collector_instance = None


class DataCollector:
    """
    数据收集器
    
    负责收集Agent执行数据、性能指标和用户反馈，
    支持数据存储和分析。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据收集器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.data_dir = self.config.get("data_dir", "data/collected")
        self.collected_data: List[Dict[str, Any]] = []
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"数据收集器初始化完成，数据目录: {self.data_dir}")
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置"""
        self.config = config
    
    async def collect(self, data: Dict[str, Any]):
        """
        收集数据
        
        Args:
            data: 要收集的数据
        """
        # 添加时间戳
        data["timestamp"] = datetime.now().isoformat()
        data["collection_id"] = len(self.collected_data)
        
        self.collected_data.append(data)
        logger.debug(f"收集数据，当前数据量: {len(self.collected_data)}")
    
    async def prepare_and_remove_data(self, task_id: Optional[str] = None) -> List[str]:
        """
        准备并移除数据
        
        Args:
            task_id: 任务ID，如果提供则只处理该任务的数据
            
        Returns:
            保存的文件路径列表
        """
        if not self.collected_data:
            return []
        
        saved_files = []
        
        try:
            # 过滤数据
            if task_id:
                filtered_data = [d for d in self.collected_data if d.get("task_id") == task_id]
            else:
                filtered_data = self.collected_data.copy()
            
            if filtered_data:
                # 保存数据到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"collected_data_{timestamp}.json"
                filepath = os.path.join(self.data_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(filepath)
                logger.info(f"保存收集数据到: {filepath}, 数据量: {len(filtered_data)}")
            
            # 清理已处理的数据
            if task_id:
                self.collected_data = [d for d in self.collected_data if d.get("task_id") != task_id]
            else:
                self.collected_data.clear()
            
            return saved_files
            
        except Exception as e:
            logger.error(f"准备和移除数据失败: {e}")
            return []
    
    def prepare_database_data(self, task_info: Dict[str, Any]) -> Optional[str]:
        """
        准备数据库数据
        
        Args:
            task_info: 任务信息
            
        Returns:
            数据库文件路径，如果失败则返回None
        """
        try:
            # 提取关键信息用于数据库存储
            db_data = {
                "op_name": task_info.get("op_name"),
                "task_desc": task_info.get("task_desc"),
                "framework": task_info.get("framework"),
                "backend": task_info.get("backend"),
                "arch": task_info.get("arch"),
                "success": task_info.get("verifier_result", False),
                "performance_metrics": task_info.get("performance_metrics"),
                "generated_code": task_info.get("coder_result"),
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存到数据库文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"database_data_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(db_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存数据库数据到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"准备数据库数据失败: {e}")
            return None
    
    def get_collected_count(self) -> int:
        """获取已收集的数据数量"""
        return len(self.collected_data)
    
    def clear_data(self):
        """清空收集的数据"""
        self.collected_data.clear()
        logger.info("清空收集的数据")


async def get_collector() -> DataCollector:
    """
    获取全局数据收集器实例
    
    Returns:
        数据收集器实例
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = DataCollector()
    return _collector_instance