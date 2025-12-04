# 从单 worker 三 Agent 到多 worker 进化式搜索架构的重构指南

> 目标：在保留现有 `analysis_chain / generation_chain / validation_chain` 的基础上，引入  
> - 多 worker 并行  
> - 策略集合 S  
> - 阈值 (UB/LB/DT) 与经验池  
> - 硬件–算子知识库  
> 使系统实现你之前设计的公式与 2.1.2 / 2.1.3 中描述的多任务并行 + 进化式搜索流程。

> 使用方式：  
> - 将本文件保存为例如 `docs/AI_REFACTOR_GUIDE.md`。  
> - 在 AI IDE（如 Cursor / Copilot Chat / 通义灵码）中引用本文件，让 AI “按这个设计重构/生成代码”。

---

## 0. 当前架构说明（供 AI 理解）

当前项目已经具备：

- 三个 Agent / Chain：
  - `analysis_chain`  
    - 输入：算子描述（来自 KernelBench / PyTorch 模型）+ 硬件信息  
    - 输出：结构化分析结果（算子类型 K、张量形状、初始设计建议等）
  - `generation_chain`  
    - 输入：分析结果 + prompt  
    - 输出：一个 Triton kernel（源码字符串 / 代码对象）
  - `validation_chain`  
    - 输入：Triton kernel + 测试输入  
    - 输出：正确性结果（pass/fail）以及性能数据（T_torch, T_kernel）

- 当前为 **单 worker 串行迭代**：每轮执行一次 `analysis -> generation -> validation`，有性能验证和报错重试机制。

希望重构为：

> 在 `analysis_chain / generation_chain / validation_chain` 之上，增加一个 **Search Orchestrator**，支持：  
> - 多 worker 并行搜索，不同 worker 使用不同“策略”  
> - 基于 speedup 的奖励与 per-strategy 最优记录  
> - UB / LB / DT 阈值控制 + 经验池 + 策略融合  
> - 硬件–算子知识库，长期保存某类算子的“好策略”。

---

## 1. 模块与文件结构规划

请 AI IDE 按下列结构创建/扩展模块（以 Python 为例，路径可按项目实际调整）：

- `core/types.py`  
  - 定义：`OperatorKey`、`HardwareSignature`、`StrategyId`、`WorkerId`。
- `core/strategy_registry.py`  
  - 定义策略集合 `ALL_STRATEGIES` 以及每个策略的提示片段 `STRATEGY_HINTS`（用于拼进 prompt）。
- `core/strategy_stats.py`  
  - `StrategyStats`：维护某策略的长期表现统计与质量评分 Q_s。
- `core/knowledge_base.py`  
  - `KnowledgeBase`：按 `(OperatorKey, HardwareSignature, StrategyId)` 组织 `StrategyStats`。
- `core/experience_pool.py`  
  - `ExperienceEntry` + `ExperiencePool`：保存高质量经验，支持“经验融合”产生新策略。
- `core/search_orchestrator.py`  
  - **核心**：多 worker 搜索主循环；负责：
    - 调度多个 worker 并行调用 `analysis_chain / generation_chain / validation_chain`；
    - 更新 per-strategy 最优记录；
    - 调用 `KnowledgeBase` 与 `ExperiencePool`。
- `core/config.py`  
  - 统一配置 UB / LB / DT、worker 数 N、Top-K 等超参数。

> 给 AI IDE 示例指令：  
> 「请根据 `AI_REFACTOR_GUIDE.md` 第 1 节的模块规划，在项目中创建这些文件和空类定义，添加合适的类型注解与 docstring。」

---

## 2. 基础类型与统计结构

### 2.1 `OperatorKey` / `HardwareSignature` / `StrategyId`

在 `core/types.py` 中实现：

```python
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass(frozen=True)
class OperatorKey:
    op_type: str                  # e.g. "matmul", "conv2d", "reduce"
    dtypes: Tuple[str, ...]       # e.g. ("fp16", "fp16")
    shape_bucket: Tuple[int, ...] # e.g. (M_bucket, N_bucket, K_bucket)

@dataclass(frozen=True)
class HardwareSignature:
    device_name: str
    sm_count: int
    shared_mem_per_sm: int
    regs_per_sm: int
    mem_bandwidth_gbps: float
    tensor_core_support: bool
    extra: Dict[str, Any] | None = None

StrategyId = str
WorkerId = str
```

### 2.2 `StrategyStats`：长期统计与 Q_s

在 `core/strategy_stats.py` 中实现一个 `StrategyStats` 类，用于维护策略 s 的长期表现：

记 speedup 样本集为 D_s，每个样本有加速比 σ_i 和成功标记。

- 平均 speedup：

  \[
  \mu_s = \frac{1}{|D_s|}\sum_{i\in D_s}\sigma_i
  \]

- speedup 方差：

  \[
  v_s = \frac{1}{|D_s|}\sum_{i\in D_s}(\sigma_i - \mu_s)^2
  \]

- 成功率：

  \[
  p_s^{succ} = \frac{\#\{\text{成功实验}\}}{|D_s|}
  \]

- 覆盖形状/配置数量：

  \[
  c_s = \#\{\text{不同形状/配置 bucket}\}
  \]

- 长期质量评分：

  \[
  Q_s = 
  \mu_s
  - \alpha\sqrt{v_s}
  + \beta p_s^{succ}
  + \gamma \log(1 + c_s)
  \]

示例接口：

```python
from dataclasses import dataclass, field
from typing import Tuple, Set
from .types import StrategyId

@dataclass
class StrategyStats:
    strategy_id: StrategyId
    samples: int = 0
    sum_sigma: float = 0.0
    sum_sigma_sq: float = 0.0
    success_count: int = 0
    shape_buckets: Set[Tuple[int, ...]] = field(default_factory=set)

    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1

    def add_sample(self, sigma: float, success: bool, shape_bucket: Tuple[int, ...]) -> None:
        ...

    @property
    def mu(self) -> float: ...
    @property
    def var(self) -> float: ...
    @property
    def p_success(self) -> float: ...
    @property
    def coverage(self) -> int: ...
    @property
    def quality(self) -> float: ...
```

> 对 AI IDE 说明：**严格按上述公式实现 `quality` 访问器**，并在 `add_sample` 中更新统计量。

---

## 3. 策略集合与 prompt 集成

### 3.1 策略注册表

在 `core/strategy_registry.py` 中维护策略集合与策略提示文本：

```python
from typing import Dict, List
from .types import StrategyId

ALL_STRATEGIES: List[StrategyId] = [
    "Triton.TileOnly.v1",
    "Triton.TileVectorize.v1",
    "Triton.ReduceOpt.v1",
    # TODO: 按需扩展
]

STRATEGY_HINTS: Dict[StrategyId, str] = {
    "Triton.TileOnly.v1": "侧重选择合适的 BLOCK_M/BLOCK_N/BLOCK_K 和 num_warps。",
    "Triton.TileVectorize.v1": "在分块基础上尽量使用向量化 load/store。",
    "Triton.ReduceOpt.v1": "优化归约和访存模式，减少非 coalesced 访问。",
}
```

### 3.2 与 `generation_chain` prompt 的对接

- 修改 `generation_chain` 的调用签名，从：

  ```python
  generation_chain(analysis_output)
  ```

  改为：

  ```python
  generation_chain(analysis_output, strategy_hint: str)
  ```

- 在 prompt 模板里增加一个 `{strategy_hint}` 占位符，用来注入不同策略的自然语言描述。

> 给 AI IDE 的具体任务：  
> 「请在 `generation_chain` 的 prompt 模板中加入 `{strategy_hint}`，并在 orchestrator 调用时从 `STRATEGY_HINTS[strategy_id]` 注入对应描述。」

---

## 4. 硬件–算子知识库 `KnowledgeBase`

在 `core/knowledge_base.py` 中实现：

### 4.1 主要职责

- 按 `(OperatorKey, HardwareSignature, StrategyId)` 维护 `StrategyStats`。  
- 提供：
  - `record_observation(op_key, hw, strategy_id, sigma, success)`  
  - `get_top_strategies(op_key, hw, k, min_samples)`  

### 4.2 接口草图

```python
from typing import Dict, Tuple, List
from .types import OperatorKey, HardwareSignature, StrategyId
from .strategy_stats import StrategyStats

class KnowledgeBase:
    def __init__(self):
        self._table: Dict[Tuple[OperatorKey, HardwareSignature, StrategyId], StrategyStats] = {}

    def record_observation(
        self,
        op_key: OperatorKey,
        hw: HardwareSignature,
        strategy_id: StrategyId,
        sigma: float,
        success: bool,
        shape_bucket: tuple[int, ...],
    ) -> None:
        ...

    def get_top_strategies(
        self,
        op_key: OperatorKey,
        hw: HardwareSignature,
        k: int = 3,
        min_samples: int = 3,
    ) -> List[StrategyStats]:
        # 按 Q_s 从高到低排序，返回适合该 (op, hw) 的策略
        ...

    def save(self, path: str) -> None:
        ...
    def load(self, path: str) -> None:
        ...
```

> 指令示例：  
> 「请在 `KnowledgeBase` 中实现 `record_observation` 和 `get_top_strategies`，内部复用 `StrategyStats`，并按 `quality` 字段排序。」

---

## 5. 经验池与策略融合

在 `core/experience_pool.py` 中实现：

### 5.1 经验条目定义

```python
from dataclasses import dataclass
from typing import Any
from .types import OperatorKey, HardwareSignature, StrategyId

@dataclass
class ExperienceEntry:
    op_key: OperatorKey
    hw: HardwareSignature
    strategy_id: StrategyId
    best_kernel_repr: Any  # 例如 Triton kernel 源码
    best_sigma: float
    UB: float
    LB: float
    DT: float
```

### 5.2 经验池与简单融合逻辑

```python
from typing import List, Optional, Tuple

class ExperiencePool:
    def __init__(self):
        self._entries: List[ExperienceEntry] = []

    def add(self, entry: ExperienceEntry) -> None:
        ...

    def size(self) -> int:
        return len(self._entries)

    def sample_two(self) -> Optional[Tuple[ExperienceEntry, ExperienceEntry]]:
        ...

    def fuse_to_new_strategy(
        self,
        e_a: ExperienceEntry,
        e_b: ExperienceEntry,
    ) -> Tuple[StrategyId, float, float, float]:
        """生成一个新策略的 StrategyId 以及新的 UB/LB/DT。
        简单实现：新ID = FUSE(a+b)，UB_new = max(UB_a, UB_b)，
        LB_new/DT_new 为二者均值。
        """
        ...
```

> 指令示例：  
> 「请在 `ExperiencePool` 中实现上述方法，并提供一个 `maybe_spawn_fused_strategy()` 辅助函数，当池中条目数 ≥ 2 时返回新策略配置，否则返回 None。」

---

## 6. SearchOrchestrator：多 worker 驱动三条 chain

在 `core/search_orchestrator.py` 中实现一个 orchestrator 类，负责：

- 调度多个 worker（每个 worker 绑定一个 StrategyId）  
- 调用 `analysis_chain / generation_chain / validation_chain`  
- 计算 speedup 和奖励  
- 更新 per-strategy 最优记录  
- 与 `KnowledgeBase`、`ExperiencePool` 交互。

### 6.1 speedup 与奖励公式

- baseline 时间：`T_torch`（由 `validation_chain` 返回）  
- 当前内核时间：`T_kernel`  
- speedup：

  ```python
  sigma = T_torch / T_kernel
  ```

- 奖励：

  ```python
  if not valid:
      reward = -1
  else:
      reward = sigma
  ```

### 6.2 per-strategy 最优记录

在 orchestrator 内维护：

```python
best_sigma: Dict[StrategyId, float]
best_kernel: Dict[StrategyId, Any]
```

更新规则：

```python
if sigma > best_sigma.get(strategy_id, 0.0):
    best_sigma[strategy_id] = sigma
    best_kernel[strategy_id] = kernel_code
```

### 6.3 单 worker 调用三条 chain

伪代码示例：

```python
from .strategy_registry import STRATEGY_HINTS

class SearchOrchestrator:
    def __init__(self, analysis_chain, generation_chain, validation_chain, kb, exp_pool, config):
        self.analysis_chain = analysis_chain
        self.generation_chain = generation_chain
        self.validation_chain = validation_chain
        self.kb = kb
        self.exp_pool = exp_pool
        self.config = config
        self.best_sigma = {}
        self.best_kernel = {}

    def run_worker_once(self, op_key, hw, strategy_id, analysis_output):
        strategy_hint = STRATEGY_HINTS[strategy_id]

        # 1) 调 generation_chain
        kernel_code = self.generation_chain(analysis_output, strategy_hint=strategy_hint)

        # 2) 调 validation_chain
        valid, T_torch, T_kernel, shape_bucket = self.validation_chain(kernel_code, ...)

        # 3) 算 sigma / reward
        if not valid:
            sigma = 0.0
            reward = -1
        else:
            sigma = T_torch / T_kernel
            reward = sigma

        # 4) 更新 per-strategy 最优
        self._update_best(strategy_id, kernel_code, sigma)

        # 5) 写回知识库
        self.kb.record_observation(op_key, hw, strategy_id, sigma, success=valid, shape_bucket=shape_bucket)

        # 6) 返回结果供阈值判断与经验池使用
        return {
            "strategy_id": strategy_id,
            "kernel_code": kernel_code,
            "sigma": sigma,
            "reward": reward,
            "valid": valid,
        }
```

---

## 7. 主搜索循环：warmup + Top/随机 + 阈值 + 经验池 + 知识库

在 `SearchOrchestrator` 中实现例如：

```python
def search_best_kernel(
    self,
    op_key: OperatorKey,
    hw: HardwareSignature,
    max_rounds: int,
    N_workers: int,
    UB: float,
    LB: float,
    DT: float,
):
    ...
```

### 7.1 warmup（第 1 轮）

- 运行一次 `analysis_chain` 得到 `analysis_output`（所有 worker 共用）。  
- 从 `ALL_STRATEGIES` 中 **随机** 选出 `N_workers` 个策略。  
- 对每个策略调用 `run_worker_once`。

### 7.2 后续轮（第 t ≥ 2）

- 从 `KnowledgeBase.get_top_strategies(op_key, hw, k)` 拿到一部分长期表现好的策略；  
- 按当前 `best_sigma` 排序，选 TOP `N/2`；  
- 从全集或剩余策略中随机采样 `N/2`；  
- 组合成 `W^(t)`，对每个策略运行 worker。

### 7.3 阈值逻辑与经验池

对策略 s 在第 t 轮结果记为 σ_s^(t)，历史最优为 σ_s^*：

- 退化量：

  \[
  \Delta_s^{(t)} = \sigma_s^* - \sigma_s^{(t)}
  \]

- 人工介入条件：
  - `1 < sigma_s_t < LB`  
  - 或 `Delta_s_t > DT`

- 写入经验池条件：
  - `sigma_s_t >= UB`

当写入经验池后：

- 把 `(op_key, hw, strategy_id, kernel_code, sigma_s_t, UB, LB, DT)` 封装为 `ExperienceEntry`，调用 `exp_pool.add(entry)`；  
- 若 `exp_pool.size() >= 2`，调用 `exp_pool.fuse_to_new_strategy(...)` 得到新策略 ID 与新阈值，并将该策略加入下一轮候选。

> 给 AI IDE 的指令示例：  
> 「请在 `SearchOrchestrator.search_best_kernel` 中实现上述 warmup / Top+随机 / 阈值判定 / 经验池融合的主循环逻辑，并在每次成功评估后更新 `best_sigma` 与 `KnowledgeBase`。」

---

## 8. 给 AI IDE 的一条总指令（可直接复制）

> 指令：  
> 请阅读 `docs/AI_REFACTOR_GUIDE.md`，并按照其中的设计对现有的 `analysis_chain / generation_chain / validation_chain` 框架进行重构：  
> 1. 保持三个 chain 内部逻辑不变，仅修改其调用方式（在 `generation_chain` 中加入策略提示参数）。  
> 2. 新增 `core/types.py`、`core/strategy_registry.py`、`core/strategy_stats.py`、`core/knowledge_base.py`、`core/experience_pool.py` 与 `core/search_orchestrator.py` 模块，实现文档中描述的数据结构与接口。  
> 3. 在 `SearchOrchestrator` 中实现多 worker 搜索主循环，包括：  
>    - warmup 随机策略；  
>    - 后续轮 Top N/2 策略 + Random N/2 策略；  
>    - 使用 speedup 作为 reward，维护 per-strategy 的最优记录 best_sigma / best_kernel；  
>    - 基于 UB/LB/DT 的人工介入判定与经验池写入逻辑；  
>    - 利用 `KnowledgeBase` 提供的长期评分 Q_s 作为策略先验；  
>    - 利用 `ExperiencePool` 对高质量经验进行策略融合并派生新 worker。  
> 4. 请在生成的代码中保持清晰的类型注解与 docstring，并与现有项目风格保持一致。
