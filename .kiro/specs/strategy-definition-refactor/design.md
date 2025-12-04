# Design Document

## Overview

本设计文档描述了将KernelGen策略系统重构为两层架构的详细设计方案。核心思想是将策略的静态定义（配置、提示、约束）与动态性能数据（运行时统计）分离，实现更灵活的策略管理和更高效的性能分析。

### 设计目标

1. **配置与数据分离**: 静态策略定义存储在`configs/strategy_definitions.json`，动态性能数据存储在`data/strategy_performance.json`
2. **人机可读**: 两个JSON文件都采用人类可读格式，支持版本控制和手动编辑
3. **向后兼容**: 支持现有的策略ID格式（如"Triton.TileVectorize.v1"），同时引入新格式（如"matmul.tile_vec.v1"）
4. **无缝集成**: 与现有的SearchOrchestrator、GenerationChain、KnowledgeBase等组件无缝集成
5. **热重载支持**: 支持运行时重新加载策略定义而无需重启

## Architecture

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Strategy System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │  Static Layer        │      │  Dynamic Layer       │    │
│  │                      │      │                      │    │
│  │  ┌────────────────┐ │      │  ┌────────────────┐ │    │
│  │  │ Strategy       │ │      │  │ Performance    │ │    │
│  │  │ Definitions    │ │      │  │ Data           │ │    │
│  │  │ (JSON)         │ │      │  │ (JSON)         │ │    │
│  │  └────────────────┘ │      │  └────────────────┘ │    │
│  │         ↓            │      │         ↓            │    │
│  │  ┌────────────────┐ │      │  ┌────────────────┐ │    │
│  │  │ Strategy       │ │      │  │ Knowledge      │ │    │
│  │  │ Registry       │ │      │  │ Base           │ │    │
│  │  └────────────────┘ │      │  └────────────────┘ │    │
│  └──────────────────────┘      └──────────────────────┘    │
│              ↓                            ↓                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Strategy Recommendation Engine             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │      Search Orchestrator             │
        │  ┌────────────┐  ┌────────────────┐ │
        │  │  Worker 0  │  │  Worker 1...N  │ │
        │  └────────────┘  └────────────────┘ │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │      Generation Chain                │
        │  (使用策略的prompt_hint和配置)       │
        └──────────────────────────────────────┘
```

### 数据流

1. **启动阶段**: StrategyRegistry从`configs/strategy_definitions.json`加载静态定义
2. **启动阶段**: KnowledgeBase从`data/strategy_performance.json`加载历史性能数据
3. **策略选择**: SearchOrchestrator查询StrategyRegistry获取适用策略，查询KnowledgeBase获取性能排名
4. **代码生成**: GenerationChain使用策略的prompt_hint、triton_preferences和constraints生成kernel
5. **性能记录**: 执行结果通过KnowledgeBase记录到动态性能数据
6. **持久化**: KnowledgeBase定期将性能数据保存到`data/strategy_performance.json`

## Components and Interfaces

### 1. Static Strategy Definition (JSON Schema)

```json
{
  "strategy_id": "matmul.tile_vec.v1",
  "op_types": ["matmul", "batch_matmul"],
  "prompt_hint": "## 策略: 分块+向量化...",
  "triton_preferences": {
    "block_sizes": {
      "BLOCK_M": [64, 128, 256],
      "BLOCK_N": [64, 128, 256],
      "BLOCK_K": [32, 64]
    },
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
    "enable_vectorization": true,
    "prefer_tensor_cores": false
  },
  "constraints": {
    "min_shape": [32, 32, 32],
    "max_shape": null,
    "supported_dtypes": ["fp16", "fp32", "bf16"],
    "required_hardware_features": [],
    "min_sm_count": 0,
    "min_shared_mem_per_sm": 0
  },
  "version": "v1",
  "metadata": {
    "description": "矩阵乘法的分块+向量化优化策略",
    "tags": ["matmul", "tiling", "vectorization"],
    "deprecated": false,
    "superseded_by": null
  }
}
```

**字段说明**:
- `strategy_id`: 唯一标识符，格式为"{op_type}.{method}.{version}"
- `op_types`: 适用的算子类型列表，空列表表示通用策略
- `prompt_hint`: 提供给LLM的策略提示文本（Markdown格式）
- `triton_preferences`: Triton编译器偏好设置
- `constraints`: 策略适用的约束条件
- `version`: 策略版本号
- `metadata`: 元数据信息

### 2. Dynamic Performance Data (JSON Schema)

```json
{
  "version": "1.0.0",
  "entries": [
    {
      "op_key": {
        "op_type": "matmul",
        "dtypes": ["fp16", "fp16"],
        "shape_bucket": [128, 128, 128]
      },
      "hardware": {
        "device_name": "NVIDIA A100",
        "sm_count": 108,
        "shared_mem_per_sm": 167936,
        "regs_per_sm": 65536,
        "mem_bandwidth_gbps": 1555.0,
        "tensor_core_support": true
      },
      "strategy_id": "matmul.tile_vec.v1",
      "stats": {
        "strategy_id": "matmul.tile_vec.v1",
        "samples": 15,
        "sum_sigma": 45.5,
        "sum_sigma_sq": 142.3,
        "success_count": 14,
        "shape_buckets": [[128, 128, 128], [256, 256, 256]],
        "alpha": 0.5,
        "beta": 0.5,
        "gamma": 0.1,
        "mu": 3.03,
        "var": 0.12,
        "std": 0.35,
        "p_success": 0.93,
        "coverage": 2,
        "quality": 3.45
      }
    }
  ]
}
```

**字段说明**:
- `version`: 性能数据格式版本
- `entries`: 性能记录列表
  - `op_key`: 算子标识符（op_type, dtypes, shape_bucket）
  - `hardware`: 硬件签名
  - `strategy_id`: 策略ID
  - `stats`: StrategyStats对象的序列化形式

### 3. StrategyDefinition Class

```python
@dataclass
class StrategyDefinition:
    """静态策略定义"""
    strategy_id: StrategyId
    op_types: List[str]
    prompt_hint: str
    triton_preferences: Dict[str, Any]
    constraints: Dict[str, Any]
    version: str
    metadata: Dict[str, Any]
    
    def matches_operator(self, op_type: str) -> bool:
        """检查策略是否适用于指定算子类型"""
        if not self.op_types:  # 空列表表示通用策略
            return True
        return op_type in self.op_types
    
    def satisfies_constraints(self, 
                            shape: Tuple[int, ...],
                            dtypes: Tuple[str, ...],
                            hardware: HardwareSignature) -> bool:
        """检查是否满足约束条件"""
        # 实现约束检查逻辑
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyDefinition":
        """从字典反序列化"""
        pass
```

### 4. Enhanced StrategyRegistry

```python
class StrategyRegistry:
    """增强的策略注册表"""
    
    def __init__(self, config_path: str = "configs/strategy_definitions.json"):
        self._config_path = config_path
        self._strategies: Dict[StrategyId, StrategyDefinition] = {}
        self._legacy_mapping: Dict[StrategyId, StrategyId] = {}
        self.load()
    
    def load(self) -> None:
        """从JSON文件加载策略定义"""
        pass
    
    def reload(self) -> None:
        """重新加载策略定义"""
        pass
    
    def get_definition(self, strategy_id: StrategyId) -> Optional[StrategyDefinition]:
        """获取策略定义"""
        pass
    
    def get_strategies_for_operator(self, op_type: str) -> List[StrategyDefinition]:
        """获取适用于指定算子的所有策略"""
        pass
    
    def get_prompt_hint(self, strategy_id: StrategyId) -> str:
        """获取策略提示（向后兼容）"""
        pass
    
    def register_legacy_mapping(self, old_id: StrategyId, new_id: StrategyId):
        """注册旧ID到新ID的映射"""
        pass
    
    def normalize_strategy_id(self, strategy_id: StrategyId) -> StrategyId:
        """规范化策略ID（处理新旧格式）"""
        pass
```

### 5. Enhanced KnowledgeBase

```python
class KnowledgeBase:
    """增强的知识库"""
    
    def __init__(self, data_path: str = "data/strategy_performance.json"):
        self._data_path = data_path
        self._table: Dict[Tuple[OperatorKey, HardwareSignature, StrategyId], StrategyStats] = {}
        self.load()
    
    def load(self) -> None:
        """从JSON文件加载性能数据"""
        pass
    
    def save(self) -> None:
        """保存性能数据到JSON文件"""
        pass
    
    def record_observation(self, ...):
        """记录观察结果（保持现有接口）"""
        pass
    
    def get_top_strategies(self, ...) -> List[StrategyStats]:
        """获取top-k策略（保持现有接口）"""
        pass
    
    def export_performance_report(self, 
                                  op_type: str,
                                  hardware: HardwareSignature,
                                  output_path: str):
        """导出性能对比报告"""
        pass
```

### 6. Strategy Recommendation Engine

```python
class StrategyRecommendationEngine:
    """策略推荐引擎"""
    
    def __init__(self, 
                 registry: StrategyRegistry,
                 knowledge_base: KnowledgeBase):
        self.registry = registry
        self.knowledge_base = knowledge_base
    
    def recommend_strategies(self,
                           op_key: OperatorKey,
                           hardware: HardwareSignature,
                           shape: Tuple[int, ...],
                           k: int = 3,
                           min_samples: int = 3) -> List[StrategyDefinition]:
        """
        推荐策略
        
        流程:
        1. 从registry获取适用于op_type的所有策略
        2. 过滤满足constraints的策略
        3. 从knowledge_base查询性能统计
        4. 按质量分数排序，返回top-k
        5. 如果历史数据不足，返回基础策略列表
        """
        pass
    
    def get_strategy_with_config(self, strategy_id: StrategyId) -> Dict[str, Any]:
        """
        获取策略及其完整配置
        
        返回:
        {
            "strategy_id": "...",
            "prompt_hint": "...",
            "triton_preferences": {...},
            "constraints": {...},
            "performance_stats": {...}  # 如果有历史数据
        }
        """
        pass
```

## Data Models

### StrategyDefinition

完整的静态策略定义数据模型，包含所有配置信息。

### StrategyStats

保持现有的StrategyStats类不变，继续用于动态性能统计。

### OperatorKey, HardwareSignature

保持现有定义不变，确保向后兼容。

## Error Handling

### 配置文件错误

1. **文件不存在**: 使用默认策略列表，记录警告日志
2. **JSON格式错误**: 抛出异常，阻止系统启动
3. **字段缺失**: 使用默认值填充，记录警告
4. **约束验证失败**: 跳过该策略，记录错误

### 性能数据错误

1. **文件不存在**: 创建空的知识库
2. **数据损坏**: 尝试恢复部分数据，记录错误
3. **版本不兼容**: 尝试迁移或重新初始化

### 运行时错误

1. **策略ID不存在**: 回退到通用策略
2. **约束不满足**: 过滤掉该策略
3. **重载失败**: 保持使用旧配置

## Testing Strategy

### 单元测试

1. **StrategyDefinition**: 测试序列化、约束检查、算子匹配
2. **StrategyRegistry**: 测试加载、查询、重载、ID映射
3. **KnowledgeBase**: 测试JSON持久化、性能查询、报告生成
4. **StrategyRecommendationEngine**: 测试推荐逻辑、过滤、排序

### 集成测试

1. **配置加载**: 测试从JSON文件加载完整配置
2. **策略推荐**: 测试端到端的策略推荐流程
3. **性能记录**: 测试运行时性能数据的记录和持久化
4. **热重载**: 测试运行时重新加载配置

### 兼容性测试

1. **旧格式支持**: 测试"Triton.TileVectorize.v1"格式的策略ID
2. **新格式支持**: 测试"matmul.tile_vec.v1"格式的策略ID
3. **混合使用**: 测试新旧格式混合使用的场景
4. **迁移工具**: 测试从旧格式迁移到新格式

### 性能测试

1. **加载性能**: 测试大量策略定义的加载时间
2. **查询性能**: 测试策略查询和推荐的响应时间
3. **持久化性能**: 测试性能数据保存的开销

## Integration Points

### 与SearchOrchestrator集成

```python
# 在SearchOrchestrator中
class SearchOrchestrator:
    def __init__(self, ...):
        self.registry = StrategyRegistry()
        self.knowledge_base = KnowledgeBase()
        self.recommender = StrategyRecommendationEngine(
            self.registry, 
            self.knowledge_base
        )
    
    async def run_search(self, ...):
        # 获取推荐策略
        recommended = self.recommender.recommend_strategies(
            op_key, hardware, shape, k=num_workers
        )
        
        # 为每个worker分配策略
        for worker_id, strategy_def in enumerate(recommended):
            strategy_config = self.recommender.get_strategy_with_config(
                strategy_def.strategy_id
            )
            # 传递给worker
            ...
```

### 与GenerationChain集成

```python
# 在GenerationChain中
async def generate_code(self, 
                       pytorch_code: str,
                       problem_info: Dict[str, Any],
                       architecture_design: Dict[str, Any],
                       implementation_guidance: Dict[str, Any],
                       strategy_config: Optional[Dict[str, Any]] = None):
    """
    strategy_config包含:
    - prompt_hint: 策略提示文本
    - triton_preferences: Triton偏好设置
    - constraints: 约束条件
    """
    if strategy_config:
        # 将triton_preferences注入到implementation_guidance
        implementation_guidance["triton_preferences"] = strategy_config["triton_preferences"]
        
        # 将prompt_hint添加到输入
        input_data["strategy_hint"] = strategy_config["prompt_hint"]
    
    # 调用LLM生成代码
    ...
```

### 与现有类型系统集成

保持OperatorKey、HardwareSignature、StrategyStats等类型不变，确保无缝集成。

## Migration Plan

### Phase 1: 创建新组件

1. 实现StrategyDefinition类
2. 创建configs/strategy_definitions.json
3. 实现增强的StrategyRegistry
4. 实现StrategyRecommendationEngine

### Phase 2: 迁移现有策略

1. 将ALL_STRATEGIES和STRATEGY_HINTS转换为JSON格式
2. 添加triton_preferences和constraints字段
3. 建立新旧ID映射关系

### Phase 3: 增强KnowledgeBase

1. 添加JSON持久化支持
2. 实现性能报告生成
3. 添加数据迁移工具

### Phase 4: 集成到现有系统

1. 修改SearchOrchestrator使用新的推荐引擎
2. 修改GenerationChain接收策略配置
3. 更新命令行工具和示例

### Phase 5: 测试和文档

1. 编写单元测试和集成测试
2. 更新文档和使用指南
3. 提供迁移指南

## Configuration Examples

### 示例1: 矩阵乘法专用策略

```json
{
  "strategy_id": "matmul.tile_tensorcore.v1",
  "op_types": ["matmul", "batch_matmul"],
  "prompt_hint": "## 策略: Tensor Core优化的矩阵乘法\n...",
  "triton_preferences": {
    "block_sizes": {
      "BLOCK_M": [128, 256],
      "BLOCK_N": [128, 256],
      "BLOCK_K": [32, 64]
    },
    "num_warps": [8],
    "num_stages": [3, 4],
    "enable_vectorization": true,
    "prefer_tensor_cores": true
  },
  "constraints": {
    "min_shape": [64, 64, 64],
    "supported_dtypes": ["fp16", "bf16"],
    "required_hardware_features": ["tensor_cores"]
  },
  "version": "v1",
  "metadata": {
    "created_at": "2024-12-04",
    "author": "system",
    "description": "针对Tensor Core优化的矩阵乘法策略",
    "tags": ["matmul", "tensor_core", "mixed_precision"]
  }
}
```

### 示例2: 通用归约策略

```json
{
  "strategy_id": "reduction.tree_reduce.v1",
  "op_types": [],
  "prompt_hint": "## 策略: 树状归约优化\n...",
  "triton_preferences": {
    "block_sizes": {
      "BLOCK_SIZE": [256, 512, 1024]
    },
    "num_warps": [4, 8],
    "num_stages": [2]
  },
  "constraints": {},
  "version": "v1",
  "metadata": {
    "created_at": "2024-12-04",
    "author": "system",
    "description": "通用的树状归约优化策略",
    "tags": ["reduction", "tree_reduce", "general"]
  }
}
```

## Performance Considerations

1. **配置加载**: 启动时一次性加载，缓存在内存中
2. **性能数据**: 定期批量保存，避免频繁IO
3. **策略查询**: 使用索引加速查询（按op_type建立索引）
4. **约束检查**: 延迟检查，只在需要时执行

## Security Considerations

1. **配置文件**: 只允许管理员修改configs/目录
2. **性能数据**: 定期备份，防止数据丢失
3. **输入验证**: 验证所有从JSON加载的数据
4. **权限控制**: 限制热重载功能的访问权限
