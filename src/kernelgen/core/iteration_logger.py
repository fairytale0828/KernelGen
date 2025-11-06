# -*- coding: utf-8 -*-
"""
迭代日志记录器
基于aikg项目的设计，记录每轮迭代的详细信息
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

class IterationLogger:
    """迭代日志记录器 - 记录每轮生成的详细信息"""
    
    def __init__(self, output_dir: str, level: int, problem_id: int):
        """
        初始化迭代日志记录器
        
        Args:
            output_dir: 输出目录
            level: KernelBench级别
            problem_id: 问题ID
        """
        self.level = level
        self.problem_id = problem_id
        
        # 创建输出目录结构
        self.base_dir = Path(output_dir)
        self.problem_dir = self.base_dir / f"level_{level}" / f"problem_{problem_id}"
        self.problem_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.iterations_dir = self.problem_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)
        
        self.kernels_dir = self.problem_dir / "kernels"
        self.kernels_dir.mkdir(exist_ok=True)
        
        # 初始化日志记录
        self.session_start_time = datetime.now()
        self.iterations_log = []
        self.best_iteration = None
        self.best_performance = None
        
        # 创建会话日志文件
        self.session_log_file = self.problem_dir / f"session_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"迭代日志记录器初始化: {self.problem_dir}")
    
    def log_iteration_start(self, iteration: int, agent_name: str, input_data: Dict[str, Any]) -> str:
        """
        记录迭代开始
        
        Args:
            iteration: 迭代次数
            agent_name: 当前Agent名称
            input_data: 输入数据
            
        Returns:
            迭代ID
        """
        iteration_id = f"iter_{iteration:03d}_{agent_name}_{int(time.time())}"
        
        iteration_log = {
            "iteration_id": iteration_id,
            "iteration": iteration,
            "agent_name": agent_name,
            "start_time": datetime.now().isoformat(),
            "input_data_summary": self._summarize_data(input_data),
            "status": "started"
        }
        
        # 保存到迭代目录
        iteration_file = self.iterations_dir / f"{iteration_id}.json"
        with open(iteration_file, 'w', encoding='utf-8') as f:
            json.dump(iteration_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"开始迭代 {iteration} - {agent_name}: {iteration_id}")
        return iteration_id
    
    def log_iteration_complete(self, iteration_id: str, output_data: Dict[str, Any], 
                             success: bool, error_message: str = "", 
                             performance_metrics: Dict[str, Any] = None) -> None:
        """
        记录迭代完成
        
        Args:
            iteration_id: 迭代ID
            output_data: 输出数据
            success: 是否成功
            error_message: 错误消息
            performance_metrics: 性能指标
        """
        # 读取现有的迭代日志
        iteration_file = self.iterations_dir / f"{iteration_id}.json"
        
        if iteration_file.exists():
            with open(iteration_file, 'r', encoding='utf-8') as f:
                iteration_log = json.load(f)
        else:
            iteration_log = {"iteration_id": iteration_id}
        
        # 更新日志
        iteration_log.update({
            "end_time": datetime.now().isoformat(),
            "success": success,
            "error_message": error_message,
            "output_data_summary": self._summarize_data(output_data),
            "performance_metrics": performance_metrics or {},
            "status": "completed" if success else "failed"
        })
        
        # 计算执行时间
        if "start_time" in iteration_log:
            start_time = datetime.fromisoformat(iteration_log["start_time"])
            end_time = datetime.fromisoformat(iteration_log["end_time"])
            iteration_log["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # 保存更新的日志
        with open(iteration_file, 'w', encoding='utf-8') as f:
            json.dump(iteration_log, f, indent=2, ensure_ascii=False)
        
        # 添加到会话日志
        self.iterations_log.append(iteration_log)
        
        # 如果有性能指标，检查是否是最佳结果
        if success and performance_metrics:
            self._update_best_result(iteration_log)
        
        logger.info(f"完成迭代 {iteration_id}: {'成功' if success else '失败'}")
    
    def log_generated_kernel(self, iteration: int, kernel_code: str, 
                           kernel_name: str, agent_name: str,
                           metadata: Dict[str, Any] = None) -> str:
        """
        记录生成的kernel代码
        
        Args:
            iteration: 迭代次数
            kernel_code: kernel代码
            kernel_name: kernel名称
            agent_name: 生成的Agent名称
            metadata: 元数据
            
        Returns:
            kernel文件路径
        """
        # 创建kernel文件名
        timestamp = datetime.now().strftime('%H%M%S')
        kernel_filename = f"iter_{iteration:03d}_{kernel_name}_{timestamp}.py"
        kernel_file = self.kernels_dir / kernel_filename
        
        # 保存kernel代码
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(f"# Generated by {agent_name} at {datetime.now().isoformat()}\n")
            f.write(f"# Iteration: {iteration}\n")
            f.write(f"# Kernel Name: {kernel_name}\n")
            if metadata:
                f.write(f"# Metadata: {json.dumps(metadata, ensure_ascii=False)}\n")
            f.write("\n")
            f.write(kernel_code)
        
        # 创建kernel元数据文件
        metadata_file = kernel_file.with_suffix('.json')
        kernel_metadata = {
            "iteration": iteration,
            "kernel_name": kernel_name,
            "agent_name": agent_name,
            "generated_at": datetime.now().isoformat(),
            "kernel_file": kernel_filename,
            "code_length": len(kernel_code),
            "metadata": metadata or {}
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(kernel_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存kernel代码: {kernel_file}")
        return str(kernel_file)
    
    def log_performance_result(self, iteration: int, triton_time: float, 
                             pytorch_time: float, speedup: float,
                             correctness: bool, additional_metrics: Dict[str, Any] = None) -> None:
        """
        记录性能测试结果
        
        Args:
            iteration: 迭代次数
            triton_time: Triton执行时间
            pytorch_time: PyTorch执行时间
            speedup: 加速比
            correctness: 正确性
            additional_metrics: 额外指标
        """
        performance_log = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "triton_time_ms": triton_time,
            "pytorch_time_ms": pytorch_time,
            "speedup": speedup,
            "correctness": correctness,
            "additional_metrics": additional_metrics or {}
        }
        
        # 保存性能日志
        performance_file = self.problem_dir / f"performance_iter_{iteration:03d}.json"
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"记录性能结果 - 迭代 {iteration}: 加速比 {speedup:.2f}x, 正确性 {correctness}")
    
    def save_session_summary(self) -> str:
        """
        保存会话摘要
        
        Returns:
            摘要文件路径
        """
        session_end_time = datetime.now()
        total_duration = (session_end_time - self.session_start_time).total_seconds()
        
        # 统计信息
        total_iterations = len(self.iterations_log)
        successful_iterations = len([log for log in self.iterations_log if log.get("success", False)])
        
        # 性能统计
        performance_stats = self._calculate_performance_stats()
        
        session_summary = {
            "session_info": {
                "level": self.level,
                "problem_id": self.problem_id,
                "start_time": self.session_start_time.isoformat(),
                "end_time": session_end_time.isoformat(),
                "total_duration_seconds": total_duration
            },
            "iteration_statistics": {
                "total_iterations": total_iterations,
                "successful_iterations": successful_iterations,
                "success_rate": successful_iterations / total_iterations if total_iterations > 0 else 0,
                "average_iteration_time": sum(log.get("duration_seconds", 0) for log in self.iterations_log) / total_iterations if total_iterations > 0 else 0
            },
            "performance_statistics": performance_stats,
            "best_result": self.best_iteration,
            "iterations_log": self.iterations_log
        }
        
        # 保存摘要
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            json.dump(session_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存会话摘要: {self.session_log_file}")
        return str(self.session_log_file)
    
    def get_best_kernel_path(self) -> Optional[str]:
        """获取最佳kernel的文件路径"""
        if self.best_iteration:
            iteration = self.best_iteration.get("iteration", 0)
            # 查找对应迭代的kernel文件
            kernel_files = list(self.kernels_dir.glob(f"iter_{iteration:03d}_*.py"))
            if kernel_files:
                return str(kernel_files[0])
        return None
    
    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """数据摘要，避免日志过大，但保留关键信息"""
        summary = {}
        for key, value in data.items():
            if isinstance(value, str):
                # 对于短字符串，保留完整内容；长字符串显示摘要
                if len(value) <= 100:
                    summary[key] = value
                else:
                    summary[key] = f"<string:{len(value)} chars>"
            elif isinstance(value, (list, tuple)):
                # 对于短列表，保留完整内容
                if len(value) <= 5:
                    summary[key] = value
                else:
                    summary[key] = f"<{type(value).__name__}:{len(value)} items>"
            elif isinstance(value, dict):
                # 对于字典，保留关键字段
                if len(value) <= 3:
                    summary[key] = value
                else:
                    summary[key] = f"<dict:{len(value)} keys>"
            elif isinstance(value, bool):
                # 布尔值直接保留
                summary[key] = value
            elif isinstance(value, (int, float)):
                # 数值直接保留
                summary[key] = value
            else:
                summary[key] = str(type(value).__name__)
        return summary
    
    def _update_best_result(self, iteration_log: Dict[str, Any]) -> None:
        """更新最佳结果"""
        performance_metrics = iteration_log.get("performance_metrics", {})
        speedup = performance_metrics.get("speedup", 0)
        correctness = performance_metrics.get("correctness", False)
        
        # 只有正确的结果才考虑
        if correctness and (self.best_performance is None or speedup > self.best_performance):
            self.best_performance = speedup
            self.best_iteration = iteration_log.copy()
            logger.info(f"更新最佳结果: 迭代 {iteration_log.get('iteration', 0)}, 加速比 {speedup:.2f}x")
    
    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """计算性能统计"""
        performance_data = []
        
        for log in self.iterations_log:
            metrics = log.get("performance_metrics", {})
            if metrics.get("speedup") is not None:
                performance_data.append({
                    "speedup": metrics["speedup"],
                    "correctness": metrics.get("correctness", False),
                    "triton_time": metrics.get("triton_time_ms", 0),
                    "pytorch_time": metrics.get("pytorch_time_ms", 0)
                })
        
        if not performance_data:
            return {"message": "无性能数据"}
        
        # 只考虑正确的结果
        correct_results = [data for data in performance_data if data["correctness"]]
        
        if not correct_results:
            return {"message": "无正确的性能结果"}
        
        speedups = [data["speedup"] for data in correct_results]
        
        return {
            "total_performance_tests": len(performance_data),
            "correct_results": len(correct_results),
            "correctness_rate": len(correct_results) / len(performance_data),
            "best_speedup": max(speedups),
            "average_speedup": sum(speedups) / len(speedups),
            "min_speedup": min(speedups)
        }