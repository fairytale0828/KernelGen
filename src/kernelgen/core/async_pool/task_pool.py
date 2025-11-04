"""
任务池管理 - 管理并发任务的执行
"""

import asyncio
import logging
from typing import List, Callable, Any, Tuple

logger = logging.getLogger(__name__)


class TaskPool:
    """
    任务池管理器
    
    负责管理并发任务的执行，支持任务队列和结果收集。
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        初始化任务池
        
        Args:
            max_concurrent_tasks: 最大并发任务数
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: List[asyncio.Task] = []
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        logger.info(f"任务池初始化完成，最大并发数: {max_concurrent_tasks}")
    
    def create_task(self, coro: Callable[[], Any]) -> asyncio.Task:
        """
        创建并添加任务到池中
        
        Args:
            coro: 协程函数
            
        Returns:
            创建的任务对象
        """
        async def _wrapped_task():
            async with self.semaphore:
                return await coro()
        
        task = asyncio.create_task(_wrapped_task())
        self.tasks.append(task)
        logger.debug(f"创建任务，当前任务数: {len(self.tasks)}")
        return task
    
    async def wait_all(self) -> List[Any]:
        """
        等待所有任务完成
        
        Returns:
            所有任务的结果列表
        """
        if not self.tasks:
            logger.warning("没有任务需要等待")
            return []
        
        logger.info(f"等待 {len(self.tasks)} 个任务完成...")
        
        try:
            results = await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # 统计成功和失败的任务
            success_count = 0
            error_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"任务 {i} 执行失败: {result}")
                    error_count += 1
                else:
                    success_count += 1
            
            logger.info(f"任务执行完成，成功: {success_count}, 失败: {error_count}")
            return results
            
        finally:
            # 清理任务列表
            self.tasks.clear()
    
    async def wait_any(self) -> Tuple[Any, List[asyncio.Task]]:
        """
        等待任意一个任务完成
        
        Returns:
            (完成的任务结果, 剩余未完成的任务列表)
        """
        if not self.tasks:
            raise ValueError("没有任务可以等待")
        
        done, pending = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # 获取完成的任务结果
        completed_task = done.pop()
        result = await completed_task
        
        # 更新任务列表
        self.tasks = list(pending)
        
        logger.debug(f"一个任务完成，剩余任务数: {len(self.tasks)}")
        return result, self.tasks
    
    def cancel_all(self):
        """取消所有未完成的任务"""
        cancelled_count = 0
        for task in self.tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        
        logger.info(f"取消了 {cancelled_count} 个未完成的任务")
        self.tasks.clear()
    
    def get_task_count(self) -> int:
        """获取当前任务数"""
        return len(self.tasks)
    
    def get_running_count(self) -> int:
        """获取正在运行的任务数"""
        return sum(1 for task in self.tasks if not task.done())
    
    def get_completed_count(self) -> int:
        """获取已完成的任务数"""
        return sum(1 for task in self.tasks if task.done())