"""
核心类型定义模块
定义多worker进化式搜索架构中使用的基础数据类型
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
import json
import hashlib


@dataclass(frozen=True)
class OperatorKey:
    """
    算子标识符
    
    用于唯一标识一个操作类型，包括操作名称、数据类型和形状桶。
    形状桶用于将相似大小的输入归类到同一组，以便知识库学习。
    
    Attributes:
        op_type: 操作类型，如"matmul", "conv2d", "reduce"
        dtypes: 数据类型元组，如("fp16", "fp16")
        shape_bucket: 形状桶，如(M_bucket, N_bucket, K_bucket)
    """
    op_type: str
    dtypes: Tuple[str, ...]
    shape_bucket: Tuple[int, ...]
    
    def __post_init__(self):
        """验证数据有效性"""
        if not self.op_type:
            raise ValueError("op_type不能为空")
        if not self.dtypes:
            raise ValueError("dtypes不能为空")
        if not self.shape_bucket:
            raise ValueError("shape_bucket不能为空")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "op_type": self.op_type,
            "dtypes": list(self.dtypes),
            "shape_bucket": list(self.shape_bucket)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorKey":
        """从字典创建实例"""
        return cls(
            op_type=data["op_type"],
            dtypes=tuple(data["dtypes"]),
            shape_bucket=tuple(data["shape_bucket"])
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.op_type}_{'-'.join(self.dtypes)}_{'-'.join(map(str, self.shape_bucket))}"


@dataclass(frozen=True)
class HardwareSignature:
    """
    硬件签名
    
    用于唯一标识硬件配置，包括设备规格和功能特性。
    
    Attributes:
        device_name: 设备名称，如"NVIDIA A100"
        sm_count: SM（流多处理器）数量
        shared_mem_per_sm: 每个SM的共享内存大小（字节）
        regs_per_sm: 每个SM的寄存器数量
        mem_bandwidth_gbps: 内存带宽（GB/s）
        tensor_core_support: 是否支持Tensor Core
        extra: 额外的硬件信息
    """
    device_name: str
    sm_count: int
    shared_mem_per_sm: int
    regs_per_sm: int
    mem_bandwidth_gbps: float
    tensor_core_support: bool
    extra: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证数据有效性"""
        if not self.device_name:
            raise ValueError("device_name不能为空")
        if self.sm_count <= 0:
            raise ValueError("sm_count必须大于0")
        if self.shared_mem_per_sm <= 0:
            raise ValueError("shared_mem_per_sm必须大于0")
        if self.regs_per_sm <= 0:
            raise ValueError("regs_per_sm必须大于0")
        if self.mem_bandwidth_gbps <= 0:
            raise ValueError("mem_bandwidth_gbps必须大于0")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "device_name": self.device_name,
            "sm_count": self.sm_count,
            "shared_mem_per_sm": self.shared_mem_per_sm,
            "regs_per_sm": self.regs_per_sm,
            "mem_bandwidth_gbps": self.mem_bandwidth_gbps,
            "tensor_core_support": self.tensor_core_support,
            "extra": self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareSignature":
        """从字典创建实例"""
        return cls(
            device_name=data["device_name"],
            sm_count=data["sm_count"],
            shared_mem_per_sm=data["shared_mem_per_sm"],
            regs_per_sm=data["regs_per_sm"],
            mem_bandwidth_gbps=data["mem_bandwidth_gbps"],
            tensor_core_support=data["tensor_core_support"],
            extra=data.get("extra")
        )
    
    def get_hash(self) -> str:
        """获取硬件签名的哈希值"""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.device_name}_SM{self.sm_count}_TC{int(self.tensor_core_support)}"


# 类型别名
StrategyId = str  # 策略ID，如"Triton.TileOnly.v1"
WorkerId = str    # Worker ID，如"worker_0"



@dataclass
class WorkerResult:
    """
    Worker执行结果
    
    封装单个worker执行一次策略的完整结果，包括生成的代码、
    性能指标和错误信息。
    
    Attributes:
        worker_id: Worker标识符
        strategy_id: 使用的策略ID
        kernel_code: 生成的Triton kernel代码
        sigma: 加速比 (T_torch / T_kernel)
        reward: 奖励值 (成功时为sigma，失败时为-1)
        valid: 是否通过正确性验证
        T_torch: PyTorch执行时间（秒）
        T_kernel: Triton kernel执行时间（秒）
        shape_bucket: 输入形状桶
        error_info: 错误信息（如果有）
        analysis_result: 分析链的输出结果
        generation_result: 生成链的输出结果
        validation_result: 验证链的输出结果
    """
    worker_id: WorkerId
    strategy_id: StrategyId
    kernel_code: str
    sigma: float
    reward: float
    valid: bool
    T_torch: float
    T_kernel: float
    shape_bucket: Tuple[int, ...]
    error_info: str = ""
    analysis_result: Optional[Dict[str, Any]] = None
    generation_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "worker_id": self.worker_id,
            "strategy_id": self.strategy_id,
            "kernel_code": self.kernel_code,
            "sigma": self.sigma,
            "reward": self.reward,
            "valid": self.valid,
            "T_torch": self.T_torch,
            "T_kernel": self.T_kernel,
            "shape_bucket": list(self.shape_bucket),
            "error_info": self.error_info,
            "analysis_result": self.analysis_result,
            "generation_result": self.generation_result,
            "validation_result": self.validation_result
        }
    
    def is_better_than(self, other: Optional["WorkerResult"]) -> bool:
        """
        判断当前结果是否优于另一个结果
        
        Args:
            other: 另一个WorkerResult，可以为None
            
        Returns:
            如果当前结果更好则返回True
        """
        if other is None:
            return self.valid
        
        # 首先比较有效性
        if self.valid and not other.valid:
            return True
        if not self.valid and other.valid:
            return False
        
        # 都有效或都无效时，比较加速比
        return self.sigma > other.sigma


@dataclass
class SearchState:
    """
    搜索状态
    
    跟踪整个搜索过程的状态，包括当前轮次、最佳结果和活跃策略。
    
    Attributes:
        round: 当前搜索轮次
        best_sigma: 每个策略的最佳加速比
        best_kernel: 每个策略的最佳kernel代码
        best_result: 每个策略的最佳WorkerResult
        active_strategies: 当前轮次活跃的策略列表
        fused_strategies: 融合策略及其提示文本
        total_evaluations: 总评估次数
        successful_evaluations: 成功评估次数
    """
    round: int = 0
    best_sigma: Dict[StrategyId, float] = field(default_factory=dict)
    best_kernel: Dict[StrategyId, str] = field(default_factory=dict)
    best_result: Dict[StrategyId, WorkerResult] = field(default_factory=dict)
    active_strategies: list[StrategyId] = field(default_factory=list)
    fused_strategies: Dict[StrategyId, str] = field(default_factory=dict)
    total_evaluations: int = 0
    successful_evaluations: int = 0
    
    def update_best(self, result: WorkerResult) -> bool:
        """
        更新策略的最佳结果
        
        Args:
            result: Worker执行结果
            
        Returns:
            如果更新了最佳结果则返回True
        """
        strategy_id = result.strategy_id
        current_best = self.best_sigma.get(strategy_id, 0.0)
        
        if result.valid and result.sigma > current_best:
            self.best_sigma[strategy_id] = result.sigma
            self.best_kernel[strategy_id] = result.kernel_code
            self.best_result[strategy_id] = result
            return True
        
        return False
    
    def get_global_best(self) -> Optional[WorkerResult]:
        """
        获取全局最佳结果
        
        Returns:
            所有策略中的最佳WorkerResult，如果没有则返回None
        """
        if not self.best_result:
            return None
        
        return max(self.best_result.values(), key=lambda r: r.sigma)
    
    def get_top_strategies(self, k: int) -> list[StrategyId]:
        """
        获取表现最好的k个策略
        
        Args:
            k: 返回的策略数量
            
        Returns:
            按加速比排序的策略ID列表
        """
        sorted_strategies = sorted(
            self.best_sigma.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [s[0] for s in sorted_strategies[:k]]
    
    def calculate_success_rate(self) -> float:
        """计算成功率"""
        if self.total_evaluations == 0:
            return 0.0
        return self.successful_evaluations / self.total_evaluations
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "round": self.round,
            "best_sigma": self.best_sigma,
            "best_kernel": self.best_kernel,
            "active_strategies": self.active_strategies,
            "fused_strategies": self.fused_strategies,
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "success_rate": self.calculate_success_rate()
        }
