# 多Worker进化式搜索架构使用指南

## 概述

多Worker进化式搜索架构是KernelGen的高级优化功能，通过并行执行多个优化策略并利用进化算法，自动发现最佳的Triton kernel实现。

## 核心特性

### 1. 并行策略搜索
- 同时运行多个worker，每个使用不同的优化策略
- 支持4-8个并行worker（可配置）
- 异步执行，充分利用计算资源

### 2. 智能策略选择
- **Warmup阶段**: 随机探索所有策略
- **后续轮次**: Top-K最佳策略 + 随机探索
- **知识库推荐**: 基于历史性能推荐策略

### 3. 进化式优化
- **经验池**: 存储高质量kernel实现
- **策略融合**: 自动组合成功策略生成新方法
- **质量评分**: Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)

### 4. 长期学习
- **知识库**: 跨会话保存策略性能统计
- **持久化**: 支持Pickle和JSON格式
- **增量学习**: 持续改进策略选择

## 快速开始

### 1. 配置

在`config.yaml`中添加多worker搜索配置：

```yaml
multi_worker_search:
  num_workers: 4          # 并行worker数量
  max_rounds: 10          # 最大搜索轮次
  
  thresholds:
    UB: 2.0              # 上界阈值（写入经验池）
    LB: 1.2              # 下界阈值（人工干预）
    DT: 0.5              # 退化阈值（人工干预）
  
  strategy_selection:
    top_k_ratio: 0.5     # Top-K策略比例
    knowledge_base_weight: 0.3  # 知识库推荐权重
  
  experience_pool:
    max_size: 100        # 最大经验数量
    fusion_probability: 0.1  # 融合概率
```

### 2. 基本使用

```python
import asyncio
from kernelgen.core import SearchOrchestrator, SearchConfig
from kernelgen.chains import AnalysisChain, GenerationChain, ValidationChain

# 创建配置
config = SearchConfig(
    num_workers=4,
    max_rounds=10,
    UB=2.0,
    LB=1.2,
    DT=0.5
)

# 初始化协调器
orchestrator = SearchOrchestrator(
    analysis_chain=analysis_chain,
    generation_chain=generation_chain,
    validation_chain=validation_chain,
    config=config
)

# 执行搜索
result = await orchestrator.search_best_kernel(
    problem_info=problem_info,
    pytorch_code=pytorch_code,
    pytorch_forward=pytorch_forward,
    test_inputs=test_inputs,
    init_inputs=init_inputs
)

print(f"最佳加速比: {result['best_speedup']:.2f}x")
print(f"最佳策略: {result['best_strategy']}")
```

### 3. 运行示例

```bash
python examples/multi_worker_search_example.py
```

## 预定义策略

系统包含8个预定义优化策略：

1. **Triton.TileOnly.v1**: 分块优化
2. **Triton.TileVectorize.v1**: 分块+向量化
3. **Triton.ReduceOpt.v1**: 归约优化
4. **Triton.SharedMemOpt.v1**: 共享内存优化
5. **Triton.WarpOpt.v1**: Warp级优化
6. **Triton.CoalescedAccess.v1**: 合并访问优化
7. **Triton.TensorCore.v1**: Tensor Core优化
8. **Triton.AsyncCopy.v1**: 异步拷贝优化

## 阈值说明

### UB (Upper Bound) - 上界阈值
- **默认值**: 2.0
- **含义**: 加速比达到此值时，将结果添加到经验池
- **用途**: 识别高质量实现，用于策略融合

### LB (Lower Bound) - 下界阈值
- **默认值**: 1.2
- **含义**: 加速比低于此值时触发人工干预警告
- **用途**: 识别性能不佳的策略

### DT (Degradation Threshold) - 退化阈值
- **默认值**: 0.5
- **含义**: 性能退化超过此值时触发警告
- **用途**: 检测策略性能下降

## 知识库管理

### 保存知识库

```python
# 保存为Pickle格式（推荐）
orchestrator.kb.save("knowledge_base.pkl")

# 保存为JSON格式（人类可读）
orchestrator.kb.save_json("knowledge_base.json")
```

### 加载知识库

```python
orchestrator.kb.load("knowledge_base.pkl")
```

### 导出最佳策略

```python
orchestrator.kb.export_best_strategies("best_strategies.json", top_k=10)
```

### 查询策略性能

```python
# 获取Top-K策略
top_strategies = orchestrator.kb.get_top_strategies(
    op_key=op_key,
    hw=hw,
    k=5,
    min_samples=3
)

for stats in top_strategies:
    print(f"{stats.strategy_id}: Q={stats.quality:.3f}, μ={stats.mu:.3f}")
```

## 经验池和策略融合

### 自动融合

系统会自动融合高质量经验：

```python
# 配置融合概率
config = SearchConfig(
    fusion_probability=0.1  # 10%概率触发融合
)
```

### 手动融合

```python
# 采样两个经验
pair = orchestrator.exp_pool.sample_best_two()

if pair:
    e_a, e_b = pair
    new_strategy_id, hint, UB, LB, DT = orchestrator.exp_pool.fuse_to_new_strategy(
        e_a, e_b, fusion_counter=0
    )
    
    # 注册融合策略
    orchestrator.registry.register_fused_strategy(new_strategy_id, hint)
```

## 性能调优

### 调整Worker数量

```yaml
multi_worker_search:
  num_workers: 8  # 增加并行度
```

**建议**:
- GPU内存充足: 6-8 workers
- GPU内存有限: 2-4 workers
- CPU测试: 2 workers

### 调整搜索轮次

```yaml
multi_worker_search:
  max_rounds: 20  # 更多轮次，更好的结果
```

**建议**:
- 快速原型: 5-10 rounds
- 生产优化: 15-20 rounds
- 深度搜索: 30+ rounds

### 调整策略选择

```yaml
strategy_selection:
  top_k_ratio: 0.7  # 更多exploitation
  # 或
  top_k_ratio: 0.3  # 更多exploration
```

## 监控和日志

### 日志级别

```python
import logging
logging.basicConfig(level=logging.INFO)

# 详细日志
logging.getLogger("kernelgen.core.search_orchestrator").setLevel(logging.DEBUG)
```

### 搜索进度

搜索过程中会输出：
- 每轮策略选择
- Worker执行状态
- 最佳结果更新
- 融合策略创建
- 阈值触发警告

### 结果分析

```python
result = await orchestrator.search_best_kernel(...)

print(f"总评估: {result['total_evaluations']}")
print(f"成功率: {result['success_rate']:.1%}")
print(f"尝试策略: {result['num_strategies_tried']}")
print(f"融合策略: {result['num_fused_strategies']}")

# Top策略排名
for strategy_info in result['top_strategies']:
    print(f"{strategy_info['strategy_id']}: {strategy_info['speedup']:.3f}x")
```

## 故障排除

### Worker超时

如果worker频繁超时：

```yaml
resource_management:
  worker_timeout: 300  # 增加到5分钟
```

### 内存不足

减少并行worker数量：

```yaml
multi_worker_search:
  num_workers: 2  # 减少并行度
```

### 策略性能差

调整阈值以更严格筛选：

```yaml
thresholds:
  UB: 2.5  # 提高经验池门槛
  LB: 1.5  # 提高干预门槛
```

## 高级用法

### 自定义策略

```python
from kernelgen.core import get_global_registry

registry = get_global_registry()

# 注册自定义策略
registry.register_strategy(
    strategy_id="Custom.MyStrategy.v1",
    hint="我的自定义优化策略提示..."
)
```

### 知识库合并

```python
# 加载多个知识库并合并
kb1 = KnowledgeBase()
kb1.load("kb1.pkl")

kb2 = KnowledgeBase()
kb2.load("kb2.pkl")

kb1.merge(kb2)
kb1.save("merged_kb.pkl")
```

### 批量优化

```python
for problem_id in range(40, 50):
    problem_info = db_loader.get_problem(level, problem_id)
    # ... 执行搜索
    
    # 知识库会累积学习
```

## 最佳实践

1. **从小规模开始**: 先用2-4 workers测试
2. **保存知识库**: 每次运行后保存知识库
3. **监控成功率**: 成功率<50%时调整配置
4. **利用融合**: 让系统自动发现新策略
5. **定期导出**: 导出最佳策略供分析

## 参考

- [设计文档](../.kiro/specs/multi-worker-evolutionary-search/design.md)
- [需求文档](../.kiro/specs/multi-worker-evolutionary-search/requirements.md)
- [实现状态](../IMPLEMENTATION_STATUS.md)
