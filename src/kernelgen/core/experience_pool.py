"""
经验池模块
存储高质量kernel经验并支持策略融合
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import random
import logging

from .types import OperatorKey, HardwareSignature, StrategyId

logger = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """
    经验条目
    
    存储一次成功的kernel优化经验，包括算子信息、硬件配置、
    策略、最佳kernel代码和性能指标。
    
    Attributes:
        op_key: 算子标识符
        hw: 硬件签名
        strategy_id: 使用的策略ID
        best_kernel_repr: 最佳kernel代码
        best_sigma: 最佳加速比
        UB: 上界阈值
        LB: 下界阈值
        DT: 退化阈值
    """
    op_key: OperatorKey
    hw: HardwareSignature
    strategy_id: StrategyId
    best_kernel_repr: str
    best_sigma: float
    UB: float
    LB: float
    DT: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "op_key": self.op_key.to_dict(),
            "hw": self.hw.to_dict(),
            "strategy_id": self.strategy_id,
            "best_kernel_repr": self.best_kernel_repr,
            "best_sigma": self.best_sigma,
            "UB": self.UB,
            "LB": self.LB,
            "DT": self.DT
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceEntry":
        """从字典创建实例"""
        return cls(
            op_key=OperatorKey.from_dict(data["op_key"]),
            hw=HardwareSignature.from_dict(data["hw"]),
            strategy_id=data["strategy_id"],
            best_kernel_repr=data["best_kernel_repr"],
            best_sigma=data["best_sigma"],
            UB=data["UB"],
            LB=data["LB"],
            DT=data["DT"]
        )


class ExperiencePool:
    """
    经验池
    
    管理高质量kernel经验，支持经验采样和策略融合。
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化经验池
        
        Args:
            max_size: 最大经验数量
        """
        self._entries: List[ExperienceEntry] = []
        self._max_size = max_size
        logger.info(f"初始化经验池 (max_size={max_size})")
    
    def add(self, entry: ExperienceEntry) -> None:
        """
        添加经验条目
        
        Args:
            entry: 经验条目
        """
        self._entries.append(entry)
        
        # 如果超过最大大小，移除最旧的条目
        if len(self._entries) > self._max_size:
            removed = self._entries.pop(0)
            logger.debug(f"经验池已满，移除最旧条目: {removed.strategy_id}")
        
        logger.info(
            f"添加经验: {entry.strategy_id} @ {entry.op_key.op_type}, "
            f"sigma={entry.best_sigma:.3f} (池大小: {len(self._entries)})"
        )
    
    def size(self) -> int:
        """
        获取经验池大小
        
        Returns:
            经验数量
        """
        return len(self._entries)
    
    def is_empty(self) -> bool:
        """
        检查经验池是否为空
        
        Returns:
            如果为空则返回True
        """
        return len(self._entries) == 0
    
    def sample_two(self) -> Optional[Tuple[ExperienceEntry, ExperienceEntry]]:
        """
        随机采样两个经验用于融合
        
        Returns:
            两个经验条目的元组，如果经验不足则返回None
        """
        if len(self._entries) < 2:
            logger.warning("经验池中经验不足，无法采样两个条目")
            return None
        
        sampled = random.sample(self._entries, 2)
        logger.debug(f"采样两个经验: {sampled[0].strategy_id}, {sampled[1].strategy_id}")
        return tuple(sampled)
    
    def sample_best_two(self) -> Optional[Tuple[ExperienceEntry, ExperienceEntry]]:
        """
        选择表现最好的两个经验用于融合
        
        Returns:
            两个经验条目的元组，如果经验不足则返回None
        """
        if len(self._entries) < 2:
            return None
        
        # 按加速比排序
        sorted_entries = sorted(self._entries, key=lambda e: e.best_sigma, reverse=True)
        return (sorted_entries[0], sorted_entries[1])
    
    def get_best_experience(self) -> Optional[ExperienceEntry]:
        """
        获取最佳经验
        
        Returns:
            加速比最高的经验条目，如果为空则返回None
        """
        if not self._entries:
            return None
        
        return max(self._entries, key=lambda e: e.best_sigma)
    
    def get_experiences_for_operator(self, op_key: OperatorKey) -> List[ExperienceEntry]:
        """
        获取特定算子的所有经验
        
        Args:
            op_key: 算子标识符
            
        Returns:
            经验条目列表
        """
        return [e for e in self._entries if e.op_key == op_key]
    
    def get_experiences_for_strategy(self, strategy_id: StrategyId) -> List[ExperienceEntry]:
        """
        获取特定策略的所有经验
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            经验条目列表
        """
        return [e for e in self._entries if e.strategy_id == strategy_id]
    
    def clear(self) -> None:
        """清空经验池"""
        self._entries.clear()
        logger.info("经验池已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取经验池统计信息
        
        Returns:
            统计信息字典
        """
        if not self._entries:
            return {
                "size": 0,
                "max_size": self._max_size,
                "avg_sigma": 0.0,
                "max_sigma": 0.0,
                "min_sigma": 0.0
            }
        
        sigmas = [e.best_sigma for e in self._entries]
        
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "avg_sigma": sum(sigmas) / len(sigmas),
            "max_sigma": max(sigmas),
            "min_sigma": min(sigmas),
            "strategies": len(set(e.strategy_id for e in self._entries)),
            "operators": len(set(str(e.op_key) for e in self._entries))
        }
    
    def __len__(self) -> int:
        """返回经验池大小"""
        return len(self._entries)
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (f"ExperiencePool(size={stats['size']}/{stats['max_size']}, "
                f"avg_sigma={stats['avg_sigma']:.3f})")
    
    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()
    
    def fuse_to_new_strategy(self,
                           e_a: ExperienceEntry,
                           e_b: ExperienceEntry,
                           fusion_counter: int = 0) -> Tuple[StrategyId, str, float, float, float]:
        """
        融合两个经验生成新策略
        
        将两个成功的经验融合，生成新的策略ID和组合的优化提示。
        
        Args:
            e_a: 第一个经验条目
            e_b: 第二个经验条目
            fusion_counter: 融合计数器（用于生成唯一ID）
            
        Returns:
            (新策略ID, 融合提示, UB_new, LB_new, DT_new) 元组
        """
        # 生成新策略ID
        strategy_a_short = e_a.strategy_id.split('.')[-1]  # 取最后一部分
        strategy_b_short = e_b.strategy_id.split('.')[-1]
        new_strategy_id = f"FUSED.{strategy_a_short}+{strategy_b_short}.v{fusion_counter}"
        
        # 计算新阈值
        UB_new = max(e_a.UB, e_b.UB)
        LB_new = (e_a.LB + e_b.LB) / 2
        DT_new = (e_a.DT + e_b.DT) / 2
        
        # 生成融合提示
        fusion_hint = self._generate_fusion_hint(e_a, e_b, new_strategy_id)
        
        logger.info(
            f"融合策略: {e_a.strategy_id} + {e_b.strategy_id} -> {new_strategy_id}, "
            f"UB={UB_new:.2f}, LB={LB_new:.2f}, DT={DT_new:.2f}"
        )
        
        return new_strategy_id, fusion_hint, UB_new, LB_new, DT_new
    
    def _generate_fusion_hint(self,
                             e_a: ExperienceEntry,
                             e_b: ExperienceEntry,
                             new_strategy_id: StrategyId) -> str:
        """
        生成融合策略的提示文本
        
        Args:
            e_a: 第一个经验条目
            e_b: 第二个经验条目
            new_strategy_id: 新策略ID
            
        Returns:
            融合策略的提示文本
        """
        hint = f"""
## 策略: 融合策略 ({new_strategy_id})

**核心思想**: 结合 {e_a.strategy_id} 和 {e_b.strategy_id} 的优化技术。

**父策略性能**:
- {e_a.strategy_id}: 加速比 {e_a.best_sigma:.2f}x
- {e_b.strategy_id}: 加速比 {e_b.best_sigma:.2f}x

**融合优化策略**:
1. 采用 {e_a.strategy_id} 的核心优化思想
2. 结合 {e_b.strategy_id} 的关键技术
3. 综合两种策略的最佳实践

**关键优化点**:
- 保持两个父策略的成功要素
- 避免冲突的优化技术
- 寻找协同效应以获得更好性能

**适用场景**:
- 与父策略相似的算子类型
- 需要综合多种优化技术的场景
- 追求更高性能的优化任务

**实现建议**:
- 参考父策略的成功经验
- 合理组合不同的优化技术
- 注意资源使用的平衡
"""
        return hint
    
    def maybe_spawn_fused_strategy(self,
                                   fusion_counter: int = 0,
                                   use_best: bool = False) -> Optional[Tuple[StrategyId, str, float, float, float]]:
        """
        尝试生成融合策略
        
        Args:
            fusion_counter: 融合计数器
            use_best: 是否使用最佳经验进行融合
            
        Returns:
            融合策略信息元组，如果无法融合则返回None
        """
        if len(self._entries) < 2:
            logger.debug("经验池中经验不足，无法生成融合策略")
            return None
        
        # 选择两个经验
        if use_best:
            pair = self.sample_best_two()
        else:
            pair = self.sample_two()
        
        if pair is None:
            return None
        
        e_a, e_b = pair
        return self.fuse_to_new_strategy(e_a, e_b, fusion_counter)
