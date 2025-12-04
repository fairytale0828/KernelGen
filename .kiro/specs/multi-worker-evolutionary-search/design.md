# 设计文档

## 概述

本设计文档描述了将现有单worker三Agent系统重构为多worker进化式搜索架构的技术方案。新架构在保留现有analysis_chain、generation_chain、validation_chain功能的基础上，引入并行worker执行、策略注册表、经验池、硬件-算子知识库等组件，实现基于进化算法的kernel优化搜索。

## 架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Search Orchestrator                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Worker 1      │  │   Worker 2      │  │   Worker N      │ │
│  │ Strategy: S1    │  │ Strategy: S2    │  │ Strategy: SN    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Existing Chain Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Analysis    │  │ Generation  │  │ Validation  │            │
│  │ Chain       │  │ Chain       │  │ Chain       │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Knowledge   │  │ Experience  │  │ Strategy    │            │
│  │ Base        │  │ Pool        │  │ Registry    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件关系

1. **Search Orchestrator**: 顶层协调器，管理多worker并行执行
2. **Worker**: 执行单元，每个绑定一个策略，调用三条chain
3. **Strategy Registry**: 策略注册表，维护所有可用策略及其提示
4. **Knowledge Base**: 知识库，存储策略长期性能统计
5. **Experience Pool**: 经验池，存储高质量结果用于策略融合

## 组件和接口

### 1. 核心类型定义 (src/kernelgen/core/types.py)

```python
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass(frozen=True)
class OperatorKey:
    """算子标识符"""
    op_type: str                  # 操作类型，如"matmul", "conv2d"
    dtypes: Tuple[str, ...]       # 数据类型，如("fp16", "fp16")
    shape_bucket: Tuple[int, ...] # 形状桶，如(M_bucket, N_bucket, K_bucket)

@dataclass(frozen=True)
class HardwareSignature:
    """硬件签名"""
    device_name: str              # 设备名称
    sm_count: int                 # SM数量
    shared_mem_per_sm: int        # 每SM共享内存
    regs_per_sm: int             # 每SM寄存器数
    mem_bandwidth_gbps: float     # 内存带宽
    tensor_core_support: bool     # Tensor Core支持
    extra: Dict[str, Any] | None = None

StrategyId = str  # 策略ID
WorkerId = str    # Worker ID
```

### 2. 策略统计 (src/kernelgen/core/strategy_stats.py)

```python
@dataclass
class StrategyStats:
    """策略长期性能统计"""
    strategy_id: StrategyId
    samples: int = 0
    sum_sigma: float = 0.0        # 加速比总和
    sum_sigma_sq: float = 0.0     # 加速比平方和
    success_count: int = 0        # 成功次数
    shape_buckets: Set[Tuple[int, ...]] = field(default_factory=set)
    
    # 质量评分权重参数
    alpha: float = 0.5  # 方差惩罚权重
    beta: float = 0.5   # 成功率奖励权重
    gamma: float = 0.1  # 覆盖范围奖励权重
    
    @property
    def mu(self) -> float:
        """平均加速比 μ_s"""
        return self.sum_sigma / self.samples if self.samples > 0 else 0.0
    
    @property
    def var(self) -> float:
        """加速比方差 v_s"""
        if self.samples <= 1:
            return 0.0
        mean = self.mu
        return (self.sum_sigma_sq / self.samples) - (mean * mean)
    
    @property
    def p_success(self) -> float:
        """成功率 p_s^succ"""
        return self.success_count / self.samples if self.samples > 0 else 0.0
    
    @property
    def coverage(self) -> int:
        """覆盖的形状桶数量 c_s"""
        return len(self.shape_buckets)
    
    @property
    def quality(self) -> float:
        """质量评分 Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)"""
        import math
        return (self.mu 
                - self.alpha * math.sqrt(self.var)
                + self.beta * self.p_success
                + self.gamma * math.log(1 + self.coverage))
```

### 3. 策略注册表 (src/kernelgen/core/strategy_registry.py)

```python
from typing import Dict, List
from .types import StrategyId

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

STRATEGY_HINTS: Dict[StrategyId, str] = {
    "Triton.TileOnly.v1": "侧重选择合适的BLOCK_M/BLOCK_N/BLOCK_K和num_warps，优化分块策略。",
    "Triton.TileVectorize.v1": "在分块基础上尽量使用向量化load/store，提高内存访问效率。",
    "Triton.ReduceOpt.v1": "优化归约和访存模式，减少非coalesced访问，使用高效的归约算法。",
    "Triton.SharedMemOpt.v1": "充分利用共享内存进行数据缓存和重用，减少全局内存访问。",
    "Triton.WarpOpt.v1": "优化warp级别的执行，使用warp shuffle和协作组操作。",
    "Triton.CoalescedAccess.v1": "确保内存访问模式的合并，优化全局内存访问效率。",
    "Triton.TensorCore.v1": "充分利用Tensor Core进行混合精度计算，优化矩阵乘法性能。",
    "Triton.AsyncCopy.v1": "使用异步内存拷贝和流水线技术，隐藏内存延迟。"
}
```

### 4. 知识库 (src/kernelgen/core/knowledge_base.py)

```python
class KnowledgeBase:
    """硬件-算子知识库"""
    
    def __init__(self):
        self._table: Dict[Tuple[OperatorKey, HardwareSignature, StrategyId], StrategyStats] = {}
    
    def record_observation(self,
                          op_key: OperatorKey,
                          hw: HardwareSignature, 
                          strategy_id: StrategyId,
                          sigma: float,
                          success: bool,
                          shape_bucket: Tuple[int, ...]) -> None:
        """记录观察结果"""
        key = (op_key, hw, strategy_id)
        if key not in self._table:
            self._table[key] = StrategyStats(strategy_id)
        
        self._table[key].add_sample(sigma, success, shape_bucket)
    
    def get_top_strategies(self,
                          op_key: OperatorKey,
                          hw: HardwareSignature,
                          k: int = 3,
                          min_samples: int = 3) -> List[StrategyStats]:
        """获取表现最佳的k个策略"""
        candidates = []
        for (op, hardware, strategy_id), stats in self._table.items():
            if (op == op_key and hardware == hw and 
                stats.samples >= min_samples):
                candidates.append(stats)
        
        # 按质量分数排序
        candidates.sort(key=lambda x: x.quality, reverse=True)
        return candidates[:k]
```

### 5. 经验池 (src/kernelgen/core/experience_pool.py)

```python
@dataclass
class ExperienceEntry:
    """经验条目"""
    op_key: OperatorKey
    hw: HardwareSignature
    strategy_id: StrategyId
    best_kernel_repr: str         # 最佳kernel代码
    best_sigma: float            # 最佳加速比
    UB: float                    # 上界阈值
    LB: float                    # 下界阈值
    DT: float                    # 退化阈值

class ExperiencePool:
    """经验池"""
    
    def __init__(self):
        self._entries: List[ExperienceEntry] = []
    
    def add(self, entry: ExperienceEntry) -> None:
        """添加经验条目"""
        self._entries.append(entry)
    
    def size(self) -> int:
        return len(self._entries)
    
    def sample_two(self) -> Optional[Tuple[ExperienceEntry, ExperienceEntry]]:
        """随机采样两个经验用于融合"""
        if len(self._entries) < 2:
            return None
        import random
        return tuple(random.sample(self._entries, 2))
    
    def fuse_to_new_strategy(self,
                           e_a: ExperienceEntry,
                           e_b: ExperienceEntry) -> Tuple[StrategyId, float, float, float]:
        """融合两个经验生成新策略"""
        # 生成新策略ID
        new_strategy_id = f"FUSED_{e_a.strategy_id}_{e_b.strategy_id}_{len(self._entries)}"
        
        # 计算新阈值
        UB_new = max(e_a.UB, e_b.UB)
        LB_new = (e_a.LB + e_b.LB) / 2
        DT_new = (e_a.DT + e_b.DT) / 2
        
        return new_strategy_id, UB_new, LB_new, DT_new
```

## 数据模型

### 搜索状态模型

```python
@dataclass
class SearchState:
    """搜索状态"""
    round: int                                    # 当前轮次
    best_sigma: Dict[StrategyId, float]          # 每策略最佳加速比
    best_kernel: Dict[StrategyId, str]           # 每策略最佳kernel
    active_strategies: List[StrategyId]          # 当前活跃策略
    fused_strategies: Dict[StrategyId, str]      # 融合策略及其提示
```

### Worker执行结果模型

```python
@dataclass 
class WorkerResult:
    """Worker执行结果"""
    worker_id: WorkerId
    strategy_id: StrategyId
    kernel_code: str
    sigma: float                 # 加速比
    reward: float               # 奖励值
    valid: bool                 # 是否有效
    T_torch: float              # PyTorch执行时间
    T_kernel: float             # Triton执行时间
    shape_bucket: Tuple[int, ...] # 形状桶
    error_info: str = ""        # 错误信息
```

## 错误处理

### 1. Worker执行错误

- **编译错误**: 记录错误信息，sigma=0，reward=-1
- **运行时错误**: 记录详细错误，标记为失败
- **超时错误**: 设置超时机制，自动终止长时间运行的worker

### 2. 策略融合错误

- **融合失败**: 记录失败原因，跳过该融合
- **新策略无效**: 验证融合策略的有效性

### 3. 知识库错误

- **持久化失败**: 提供内存备份机制
- **数据损坏**: 实现数据校验和恢复

## 测试策略

### 1. 单元测试

- **StrategyStats**: 测试统计计算的正确性
- **KnowledgeBase**: 测试存储和检索功能
- **ExperiencePool**: 测试融合逻辑

### 2. 集成测试

- **Worker并行执行**: 测试多worker协调
- **策略选择**: 测试Top-K和随机选择逻辑
- **阈值判断**: 测试UB/LB/DT阈值逻辑

### 3. 性能测试

- **并发性能**: 测试多worker并发执行效率
- **内存使用**: 监控知识库和经验池内存占用
- **搜索收敛**: 测试搜索算法的收敛性

## 配置扩展

### config.yaml新增配置

```yaml
# 多worker搜索配置
multi_worker_search:
  # Worker配置
  num_workers: 4                    # 并行worker数量
  max_rounds: 10                    # 最大搜索轮次
  
  # 阈值配置
  thresholds:
    UB: 2.0                        # 上界阈值
    LB: 1.2                        # 下界阈值  
    DT: 0.5                        # 退化阈值
  
  # 策略选择配置
  strategy_selection:
    warmup_random: true            # 预热阶段随机选择
    top_k_ratio: 0.5              # Top-K策略比例
    knowledge_base_weight: 0.3     # 知识库推荐权重
  
  # 经验池配置
  experience_pool:
    max_size: 100                  # 最大经验数量
    fusion_probability: 0.1        # 融合概率
  
  # 知识库配置
  knowledge_base:
    persistence_path: "knowledge_base.pkl"
    min_samples_for_recommendation: 3
    quality_score_weights:
      alpha: 0.5                   # 方差惩罚权重
      beta: 0.5                    # 成功率权重
      gamma: 0.1                   # 覆盖范围权重
```

## 部署考虑

### 1. 向后兼容性

- 保持现有API接口不变
- 支持单worker模式作为fallback
- 渐进式迁移策略

### 2. 资源管理

- GPU内存管理：避免多worker同时占用过多显存
- CPU资源调度：合理分配worker执行时间
- 磁盘I/O优化：批量持久化知识库数据

### 3. 监控和日志

- Worker执行状态监控
- 策略性能趋势分析
- 搜索收敛性监控
- 详细的执行日志记录