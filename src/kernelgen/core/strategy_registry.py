"""
策略注册表模块
维护所有可用的优化策略及其提示文本
"""

from typing import Dict, List, Optional
import logging

from .types import StrategyId

logger = logging.getLogger(__name__)


# 所有预定义的优化策略
ALL_STRATEGIES: List[StrategyId] = [
    "Triton.TileOnly.v1",
    "Triton.TileVectorize.v1",
    "Triton.ReduceOpt.v1",
    "Triton.SharedMemOpt.v1",
    "Triton.WarpOpt.v1",
    "Triton.CoalescedAccess.v1",
    "Triton.TensorCore.v1",
    "Triton.AsyncCopy.v1"
]


# 策略提示文本映射
STRATEGY_HINTS: Dict[StrategyId, str] = {
    "Triton.TileOnly.v1": """
## 策略: 分块优化 (Tile Only)

**核心思想**: 侧重选择合适的BLOCK_M/BLOCK_N/BLOCK_K和num_warps，优化分块策略。

**关键优化点**:
- 选择合适的块大小以最大化SM占用率
- 调整num_warps以平衡并行度和资源使用
- 确保块大小是2的幂次以优化内存访问
- 考虑硬件限制（共享内存、寄存器数量）

**适用场景**: 
- 规则的矩阵运算
- 内存访问模式简单的操作
- 需要快速原型验证的场景
""",

    "Triton.TileVectorize.v1": """
## 策略: 分块+向量化 (Tile + Vectorize)

**核心思想**: 在分块基础上尽量使用向量化load/store，提高内存访问效率。

**关键优化点**:
- 使用tl.load和tl.store的向量化版本
- 确保内存访问对齐以启用向量化
- 选择合适的向量宽度（通常是4或8）
- 利用Triton的自动向量化特性

**适用场景**:
- 连续内存访问模式
- 元素级操作（element-wise）
- 需要高内存带宽的操作
""",

    "Triton.ReduceOpt.v1": """
## 策略: 归约优化 (Reduction Optimization)

**核心思想**: 优化归约和访存模式，减少非coalesced访问，使用高效的归约算法。

**关键优化点**:
- 使用tl.sum、tl.max等内置归约操作
- 实现树状归约以减少同步开销
- 优化归约维度的选择
- 使用共享内存进行中间结果缓存

**适用场景**:
- Sum、Mean、Max等归约操作
- Softmax、LayerNorm等需要归约的操作
- 需要跨维度聚合的计算
""",

    "Triton.SharedMemOpt.v1": """
## 策略: 共享内存优化 (Shared Memory Optimization)

**核心思想**: 充分利用共享内存进行数据缓存和重用，减少全局内存访问。

**关键优化点**:
- 将频繁访问的数据加载到共享内存
- 实现数据重用以摊销内存访问成本
- 避免共享内存bank冲突
- 合理规划共享内存使用以提高占用率

**适用场景**:
- 矩阵乘法等需要数据重用的操作
- 卷积操作
- 需要线程间数据共享的算法
""",

    "Triton.WarpOpt.v1": """
## 策略: Warp级优化 (Warp-Level Optimization)

**核心思想**: 优化warp级别的执行，使用warp shuffle和协作组操作。

**关键优化点**:
- 确保warp内线程执行一致以避免分支发散
- 使用warp级原语进行高效通信
- 优化warp调度以隐藏延迟
- 合理设置num_warps参数

**适用场景**:
- 需要warp内通信的算法
- 对分支敏感的操作
- 需要细粒度并行控制的场景
""",

    "Triton.CoalescedAccess.v1": """
## 策略: 合并访问优化 (Coalesced Access)

**核心思想**: 确保内存访问模式的合并，优化全局内存访问效率。

**关键优化点**:
- 确保连续线程访问连续内存地址
- 使用合适的数据布局（行优先/列优先）
- 避免跨步访问和随机访问
- 利用L2缓存提高访问效率

**适用场景**:
- 大规模数据处理
- 内存带宽受限的操作
- 需要优化全局内存访问的场景
""",

    "Triton.TensorCore.v1": """
## 策略: Tensor Core优化 (Tensor Core Optimization)

**核心思想**: 充分利用Tensor Core进行混合精度计算，优化矩阵乘法性能。

**关键优化点**:
- 使用FP16或BF16数据类型
- 确保矩阵维度是Tensor Core tile大小的倍数
- 使用tl.dot操作触发Tensor Core
- 合理安排数据布局以匹配Tensor Core要求

**适用场景**:
- 矩阵乘法（GEMM）
- 卷积操作
- 支持混合精度的深度学习操作
""",

    "Triton.AsyncCopy.v1": """
## 策略: 异步拷贝优化 (Async Copy)

**核心思想**: 使用异步内存拷贝和流水线技术，隐藏内存延迟。

**关键优化点**:
- 使用异步加载指令
- 实现双缓冲或多缓冲流水线
- 重叠计算和内存传输
- 合理安排同步点以最小化等待时间

**适用场景**:
- 计算密集型操作
- 需要隐藏内存延迟的场景
- 大规模矩阵运算
"""
}


class StrategyRegistry:
    """
    策略注册表
    
    管理所有可用的优化策略，支持动态注册和查询。
    """
    
    def __init__(self):
        """初始化策略注册表"""
        self._strategies: List[StrategyId] = ALL_STRATEGIES.copy()
        self._hints: Dict[StrategyId, str] = STRATEGY_HINTS.copy()
        self._fused_strategies: Dict[StrategyId, str] = {}
    
    def register_strategy(self, strategy_id: StrategyId, hint: str) -> None:
        """
        注册新策略
        
        Args:
            strategy_id: 策略ID
            hint: 策略提示文本
        """
        if strategy_id in self._strategies:
            logger.warning(f"策略 {strategy_id} 已存在，将被覆盖")
        
        self._strategies.append(strategy_id)
        self._hints[strategy_id] = hint
        logger.info(f"注册策略: {strategy_id}")
    
    def register_fused_strategy(self, strategy_id: StrategyId, hint: str) -> None:
        """
        注册融合策略
        
        Args:
            strategy_id: 融合策略ID
            hint: 融合策略提示文本
        """
        self._fused_strategies[strategy_id] = hint
        self.register_strategy(strategy_id, hint)
        logger.info(f"注册融合策略: {strategy_id}")
    
    def get_hint(self, strategy_id: StrategyId) -> Optional[str]:
        """
        获取策略提示
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略提示文本，如果不存在则返回None
        """
        return self._hints.get(strategy_id)
    
    def get_all_strategies(self) -> List[StrategyId]:
        """
        获取所有策略
        
        Returns:
            所有策略ID列表
        """
        return self._strategies.copy()
    
    def get_base_strategies(self) -> List[StrategyId]:
        """
        获取基础策略（不包括融合策略）
        
        Returns:
            基础策略ID列表
        """
        return [s for s in self._strategies if s not in self._fused_strategies]
    
    def get_fused_strategies(self) -> List[StrategyId]:
        """
        获取融合策略
        
        Returns:
            融合策略ID列表
        """
        return list(self._fused_strategies.keys())
    
    def is_valid_strategy(self, strategy_id: StrategyId) -> bool:
        """
        检查策略是否有效
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            如果策略存在则返回True
        """
        return strategy_id in self._strategies
    
    def get_strategy_count(self) -> int:
        """
        获取策略总数
        
        Returns:
            策略数量
        """
        return len(self._strategies)
    
    def __len__(self) -> int:
        """返回策略数量"""
        return len(self._strategies)
    
    def __contains__(self, strategy_id: StrategyId) -> bool:
        """检查策略是否存在"""
        return strategy_id in self._strategies
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"StrategyRegistry(strategies={len(self._strategies)}, fused={len(self._fused_strategies)})"


# 全局策略注册表实例
_global_registry: Optional[StrategyRegistry] = None


def get_global_registry() -> StrategyRegistry:
    """
    获取全局策略注册表实例
    
    Returns:
        全局StrategyRegistry实例
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry
