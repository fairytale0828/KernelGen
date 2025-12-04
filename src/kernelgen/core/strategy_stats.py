"""
策略统计模块
维护策略的长期性能统计和质量评分
"""

from dataclasses import dataclass, field
from typing import Tuple, Set, Dict, Any
import math

from .types import StrategyId


@dataclass
class StrategyStats:
    """
    策略长期性能统计
    
    维护某个策略在特定算子-硬件组合下的长期表现统计，
    包括加速比、成功率、覆盖范围等指标，并计算综合质量评分。
    
    质量评分公式：
    Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)
    
    其中：
    - μ_s: 平均加速比
    - v_s: 加速比方差
    - p_s^succ: 成功率
    - c_s: 覆盖的形状桶数量
    - α, β, γ: 可配置的权重参数
    
    Attributes:
        strategy_id: 策略标识符
        samples: 样本数量
        sum_sigma: 加速比总和
        sum_sigma_sq: 加速比平方和
        success_count: 成功次数
        shape_buckets: 遇到的形状桶集合
        alpha: 方差惩罚权重
        beta: 成功率奖励权重
        gamma: 覆盖范围奖励权重
    """
    strategy_id: StrategyId
    samples: int = 0
    sum_sigma: float = 0.0
    sum_sigma_sq: float = 0.0
    success_count: int = 0
    shape_buckets: Set[Tuple[int, ...]] = field(default_factory=set)
    
    # 质量评分权重参数
    alpha: float = 0.5  # 方差惩罚权重
    beta: float = 0.5   # 成功率奖励权重
    gamma: float = 0.1  # 覆盖范围奖励权重
    
    def add_sample(self, sigma: float, success: bool, shape_bucket: Tuple[int, ...]) -> None:
        """
        添加一个新的观察样本
        
        Args:
            sigma: 加速比
            success: 是否成功（通过正确性验证）
            shape_bucket: 输入形状桶
        """
        self.samples += 1
        self.sum_sigma += sigma
        self.sum_sigma_sq += sigma * sigma
        
        if success:
            self.success_count += 1
        
        self.shape_buckets.add(shape_bucket)
    
    @property
    def mu(self) -> float:
        """
        平均加速比 μ_s
        
        Returns:
            平均加速比，如果没有样本则返回0
        """
        if self.samples == 0:
            return 0.0
        return self.sum_sigma / self.samples
    
    @property
    def var(self) -> float:
        """
        加速比方差 v_s
        
        使用公式: v_s = E[σ²] - (E[σ])²
        
        Returns:
            加速比方差，如果样本数<=1则返回0
        """
        if self.samples <= 1:
            return 0.0
        
        mean = self.mu
        mean_of_squares = self.sum_sigma_sq / self.samples
        return mean_of_squares - (mean * mean)
    
    @property
    def std(self) -> float:
        """
        加速比标准差
        
        Returns:
            加速比标准差
        """
        return math.sqrt(self.var)
    
    @property
    def p_success(self) -> float:
        """
        成功率 p_s^succ
        
        Returns:
            成功率，如果没有样本则返回0
        """
        if self.samples == 0:
            return 0.0
        return self.success_count / self.samples
    
    @property
    def coverage(self) -> int:
        """
        覆盖的形状桶数量 c_s
        
        Returns:
            遇到的不同形状桶数量
        """
        return len(self.shape_buckets)
    
    @property
    def quality(self) -> float:
        """
        质量评分 Q_s
        
        综合考虑平均性能、稳定性、成功率和覆盖范围的质量评分。
        
        公式: Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)
        
        Returns:
            质量评分，值越高表示策略质量越好
        """
        if self.samples == 0:
            return 0.0
        
        # 计算各个组成部分
        mean_speedup = self.mu
        variance_penalty = self.alpha * self.std
        success_reward = self.beta * self.p_success
        coverage_reward = self.gamma * math.log(1 + self.coverage)
        
        return mean_speedup - variance_penalty + success_reward + coverage_reward
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算加速比的置信区间
        
        Args:
            confidence: 置信水平，默认0.95
            
        Returns:
            (下界, 上界) 元组
        """
        if self.samples < 2:
            return (self.mu, self.mu)
        
        # 使用t分布计算置信区间
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, self.samples - 1)
        margin = t_value * self.std / math.sqrt(self.samples)
        
        return (self.mu - margin, self.mu + margin)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含所有统计信息的字典
        """
        return {
            "strategy_id": self.strategy_id,
            "samples": self.samples,
            "sum_sigma": self.sum_sigma,
            "sum_sigma_sq": self.sum_sigma_sq,
            "success_count": self.success_count,
            "shape_buckets": [list(bucket) for bucket in self.shape_buckets],
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            # 计算属性
            "mu": self.mu,
            "var": self.var,
            "std": self.std,
            "p_success": self.p_success,
            "coverage": self.coverage,
            "quality": self.quality
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyStats":
        """
        从字典创建实例
        
        Args:
            data: 包含统计信息的字典
            
        Returns:
            StrategyStats实例
        """
        stats = cls(
            strategy_id=data["strategy_id"],
            samples=data["samples"],
            sum_sigma=data["sum_sigma"],
            sum_sigma_sq=data["sum_sigma_sq"],
            success_count=data["success_count"],
            shape_buckets=set(tuple(bucket) for bucket in data["shape_buckets"]),
            alpha=data.get("alpha", 0.5),
            beta=data.get("beta", 0.5),
            gamma=data.get("gamma", 0.1)
        )
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"StrategyStats({self.strategy_id}: "
                f"samples={self.samples}, "
                f"μ={self.mu:.3f}, "
                f"σ={self.std:.3f}, "
                f"p_succ={self.p_success:.2%}, "
                f"coverage={self.coverage}, "
                f"Q={self.quality:.3f})")
    
    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()
