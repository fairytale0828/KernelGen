"""
硬件-算子知识库模块
长期存储和管理策略性能统计
"""

from typing import Dict, Tuple, List, Optional
import logging
import pickle
from pathlib import Path

from .types import OperatorKey, HardwareSignature, StrategyId
from .strategy_stats import StrategyStats

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    硬件-算子知识库
    
    按(OperatorKey, HardwareSignature, StrategyId)三元组索引存储策略性能统计，
    支持长期学习和策略推荐。
    
    Attributes:
        _table: 存储策略统计的字典
        _version: 知识库版本号
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        """初始化知识库"""
        self._table: Dict[Tuple[OperatorKey, HardwareSignature, StrategyId], StrategyStats] = {}
        self._version = self.VERSION
        logger.info("初始化知识库")
    
    def record_observation(self,
                          op_key: OperatorKey,
                          hw: HardwareSignature,
                          strategy_id: StrategyId,
                          sigma: float,
                          success: bool,
                          shape_bucket: Tuple[int, ...]) -> None:
        """
        记录一次观察结果
        
        Args:
            op_key: 算子标识符
            hw: 硬件签名
            strategy_id: 策略ID
            sigma: 加速比
            success: 是否成功
            shape_bucket: 形状桶
        """
        key = (op_key, hw, strategy_id)
        
        # 如果不存在则创建新的统计对象
        if key not in self._table:
            self._table[key] = StrategyStats(strategy_id)
            logger.debug(f"为 {strategy_id} 创建新的统计记录")
        
        # 添加样本
        self._table[key].add_sample(sigma, success, shape_bucket)
        
        logger.debug(
            f"记录观察: {strategy_id} @ {op_key.op_type}, "
            f"sigma={sigma:.3f}, success={success}"
        )
    
    def get_top_strategies(self,
                          op_key: OperatorKey,
                          hw: HardwareSignature,
                          k: int = 3,
                          min_samples: int = 3) -> List[StrategyStats]:
        """
        获取表现最佳的k个策略
        
        Args:
            op_key: 算子标识符
            hw: 硬件签名
            k: 返回的策略数量
            min_samples: 最小样本数要求
            
        Returns:
            按质量分数排序的策略统计列表
        """
        candidates = []
        
        # 收集符合条件的策略
        for (stored_op, stored_hw, strategy_id), stats in self._table.items():
            if stored_op == op_key and stored_hw == hw:
                if stats.samples >= min_samples:
                    candidates.append(stats)
        
        # 按质量分数排序
        candidates.sort(key=lambda x: x.quality, reverse=True)
        
        top_k = candidates[:k]
        
        if top_k:
            logger.info(
                f"找到 {len(top_k)} 个推荐策略 (共 {len(candidates)} 个候选) "
                f"for {op_key.op_type}"
            )
            for i, stats in enumerate(top_k, 1):
                logger.info(f"  {i}. {stats.strategy_id}: Q={stats.quality:.3f}")
        else:
            logger.info(f"没有找到符合条件的策略 for {op_key.op_type}")
        
        return top_k
    
    def get_strategy_stats(self,
                          op_key: OperatorKey,
                          hw: HardwareSignature,
                          strategy_id: StrategyId) -> Optional[StrategyStats]:
        """
        获取特定策略的统计信息
        
        Args:
            op_key: 算子标识符
            hw: 硬件签名
            strategy_id: 策略ID
            
        Returns:
            策略统计对象，如果不存在则返回None
        """
        key = (op_key, hw, strategy_id)
        return self._table.get(key)
    
    def get_all_strategies_for_operator(self,
                                       op_key: OperatorKey,
                                       hw: HardwareSignature) -> List[StrategyStats]:
        """
        获取某个算子-硬件组合的所有策略统计
        
        Args:
            op_key: 算子标识符
            hw: 硬件签名
            
        Returns:
            策略统计列表
        """
        result = []
        for (stored_op, stored_hw, strategy_id), stats in self._table.items():
            if stored_op == op_key and stored_hw == hw:
                result.append(stats)
        
        return result
    
    def get_statistics_summary(self) -> Dict[str, any]:
        """
        获取知识库统计摘要
        
        Returns:
            包含统计信息的字典
        """
        total_entries = len(self._table)
        total_samples = sum(stats.samples for stats in self._table.values())
        
        # 按算子类型统计
        op_types = {}
        for (op_key, _, _), stats in self._table.items():
            op_type = op_key.op_type
            if op_type not in op_types:
                op_types[op_type] = {"entries": 0, "samples": 0}
            op_types[op_type]["entries"] += 1
            op_types[op_type]["samples"] += stats.samples
        
        # 按策略统计
        strategies = {}
        for (_, _, strategy_id), stats in self._table.items():
            if strategy_id not in strategies:
                strategies[strategy_id] = {"entries": 0, "samples": 0, "avg_quality": 0.0}
            strategies[strategy_id]["entries"] += 1
            strategies[strategy_id]["samples"] += stats.samples
            strategies[strategy_id]["avg_quality"] += stats.quality
        
        # 计算平均质量
        for strategy_id in strategies:
            count = strategies[strategy_id]["entries"]
            strategies[strategy_id]["avg_quality"] /= count
        
        return {
            "version": self._version,
            "total_entries": total_entries,
            "total_samples": total_samples,
            "operator_types": op_types,
            "strategies": strategies
        }
    
    def clear(self) -> None:
        """清空知识库"""
        self._table.clear()
        logger.info("知识库已清空")
    
    def size(self) -> int:
        """返回知识库条目数量"""
        return len(self._table)
    
    def __len__(self) -> int:
        """返回知识库条目数量"""
        return len(self._table)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"KnowledgeBase(entries={len(self._table)}, version={self._version})"
    
    def __repr__(self) -> str:
        """详细表示"""
        summary = self.get_statistics_summary()
        return (f"KnowledgeBase(entries={summary['total_entries']}, "
                f"samples={summary['total_samples']}, "
                f"operators={len(summary['operator_types'])}, "
                f"strategies={len(summary['strategies'])})")
    
    def save(self, path: str) -> None:
        """
        保存知识库到文件
        
        Args:
            path: 保存路径
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备序列化数据
            data = {
                "version": self._version,
                "entries": []
            }
            
            # 序列化每个条目
            for (op_key, hw, strategy_id), stats in self._table.items():
                entry = {
                    "op_key": op_key.to_dict(),
                    "hw": hw.to_dict(),
                    "strategy_id": strategy_id,
                    "stats": stats.to_dict()
                }
                data["entries"].append(entry)
            
            # 保存到文件
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"知识库已保存到 {path} ({len(self._table)} 条记录)")
            
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        从文件加载知识库
        
        Args:
            path: 加载路径
        """
        try:
            load_path = Path(path)
            
            if not load_path.exists():
                logger.warning(f"知识库文件不存在: {path}")
                return
            
            # 从文件加载
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            # 检查版本兼容性
            loaded_version = data.get("version", "unknown")
            if loaded_version != self.VERSION:
                logger.warning(
                    f"知识库版本不匹配: 文件={loaded_version}, 当前={self.VERSION}"
                )
            
            # 清空当前数据
            self._table.clear()
            
            # 反序列化每个条目
            for entry in data.get("entries", []):
                op_key = OperatorKey.from_dict(entry["op_key"])
                hw = HardwareSignature.from_dict(entry["hw"])
                strategy_id = entry["strategy_id"]
                stats = StrategyStats.from_dict(entry["stats"])
                
                key = (op_key, hw, strategy_id)
                self._table[key] = stats
            
            logger.info(f"从 {path} 加载知识库 ({len(self._table)} 条记录)")
            
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            raise
    
    def save_json(self, path: str) -> None:
        """
        以JSON格式保存知识库（用于人类可读）
        
        Args:
            path: 保存路径
        """
        try:
            import json
            
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备JSON数据
            data = {
                "version": self._version,
                "entries": []
            }
            
            # 序列化每个条目
            for (op_key, hw, strategy_id), stats in self._table.items():
                entry = {
                    "op_key": op_key.to_dict(),
                    "hw": hw.to_dict(),
                    "strategy_id": strategy_id,
                    "stats": stats.to_dict()
                }
                data["entries"].append(entry)
            
            # 保存到JSON文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"知识库已保存为JSON到 {path}")
            
        except Exception as e:
            logger.error(f"保存JSON知识库失败: {e}")
            raise
    
    def merge(self, other: "KnowledgeBase") -> None:
        """
        合并另一个知识库
        
        Args:
            other: 要合并的知识库
        """
        merged_count = 0
        
        for key, other_stats in other._table.items():
            if key in self._table:
                # 如果已存在，合并统计数据
                self_stats = self._table[key]
                self_stats.samples += other_stats.samples
                self_stats.sum_sigma += other_stats.sum_sigma
                self_stats.sum_sigma_sq += other_stats.sum_sigma_sq
                self_stats.success_count += other_stats.success_count
                self_stats.shape_buckets.update(other_stats.shape_buckets)
            else:
                # 如果不存在，直接添加
                self._table[key] = other_stats
            
            merged_count += 1
        
        logger.info(f"合并知识库: {merged_count} 条记录")
    
    def export_best_strategies(self, output_path: str, top_k: int = 10) -> None:
        """
        导出表现最好的策略
        
        Args:
            output_path: 输出路径
            top_k: 导出前k个策略
        """
        try:
            import json
            
            # 收集所有策略并按质量排序
            all_stats = list(self._table.values())
            all_stats.sort(key=lambda x: x.quality, reverse=True)
            
            # 准备导出数据
            export_data = {
                "top_strategies": [],
                "generated_at": str(Path(output_path).stat().st_mtime if Path(output_path).exists() else "now")
            }
            
            for stats in all_stats[:top_k]:
                export_data["top_strategies"].append({
                    "strategy_id": stats.strategy_id,
                    "quality": stats.quality,
                    "mean_speedup": stats.mu,
                    "std_speedup": stats.std,
                    "success_rate": stats.p_success,
                    "coverage": stats.coverage,
                    "samples": stats.samples
                })
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"导出前 {top_k} 个最佳策略到 {output_path}")
            
        except Exception as e:
            logger.error(f"导出最佳策略失败: {e}")
            raise
