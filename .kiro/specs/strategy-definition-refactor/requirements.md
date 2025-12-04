# Requirements Document

## Introduction

本需求文档描述了将KernelGen的策略系统重构为两层架构的功能需求：静态策略定义层（供人类和LLM使用）和动态性能数据层（供搜索和推荐使用）。这种分离将使策略配置更易于管理，性能数据更易于分析，并支持更灵活的策略演化。

## Glossary

- **Strategy System**: 策略系统，管理优化策略的定义、注册和性能跟踪的完整系统
- **Static Strategy Definition**: 静态策略定义，存储在JSON文件中的策略配置，包含提示文本、Triton偏好设置等
- **Dynamic Performance Data**: 动态性能数据，运行时收集的策略在特定硬件和算子上的表现统计
- **Strategy Registry**: 策略注册表，管理策略定义的加载和查询
- **Knowledge Base**: 知识库，存储和管理动态性能数据
- **Operation Service**: 操作服务，使用LLM识别算子类型
- **Strategy ID**: 策略标识符，格式如"matmul.tile_vec.v1"或"Triton.TileVectorize.v1"

## Requirements

### Requirement 1

**User Story:** 作为开发者，我希望能够通过编辑JSON文件来定义和修改优化策略，以便快速调整策略行为而无需修改代码

#### Acceptance Criteria

1. WHEN 系统启动时，THE Strategy System SHALL 从configs/strategy_definitions.json文件加载所有静态策略定义
2. THE Static Strategy Definition SHALL 包含strategy_id、op_types列表、prompt_hint文本、triton_preferences字典、constraints字典、version字符串和metadata字典字段
3. THE Strategy Registry SHALL 支持通过strategy_id查询策略定义的所有字段
4. WHEN 策略定义文件被修改时，THE Strategy System SHALL 支持重新加载而无需重启应用程序
5. THE Static Strategy Definition SHALL 使用人类可读的JSON格式，包含注释说明（通过metadata字段）

### Requirement 2

**User Story:** 作为系统，我需要将策略的运行时性能数据与静态定义分离存储，以便独立管理和分析性能统计

#### Acceptance Criteria

1. THE Dynamic Performance Data SHALL 存储在data/strategy_performance.json文件中，与静态定义文件分离
2. THE Dynamic Performance Data SHALL 按op_type、shape_bucket、dtypes、hardware和strategy_id进行索引
3. THE Dynamic Performance Data SHALL 包含samples、sum_sigma、sum_sigma_sq、success_count和shape_buckets统计字段
4. WHEN 新的性能观察被记录时，THE Knowledge Base SHALL 更新对应的动态性能数据条目
5. THE Knowledge Base SHALL 支持将动态性能数据持久化为JSON格式以便人类查看和分析

### Requirement 3

**User Story:** 作为开发者，我希望策略ID能够反映算子类型和优化方法，以便快速理解策略的适用场景

#### Acceptance Criteria

1. THE Strategy ID SHALL 遵循命名约定："{op_type}.{optimization_method}.{version}"格式
2. THE Strategy System SHALL 支持两种ID格式：新格式（如"matmul.tile_vec.v1"）和旧格式（如"Triton.TileVectorize.v1"）
3. WHEN 查询策略时，THE Strategy Registry SHALL 能够通过op_types字段过滤适用于特定算子类型的策略
4. THE Strategy System SHALL 提供ID格式转换工具，支持新旧格式之间的映射
5. THE Static Strategy Definition SHALL 在metadata中记录策略的创建时间和作者信息

### Requirement 4

**User Story:** 作为搜索系统，我需要根据算子类型、硬件和历史性能快速推荐最佳策略，以便提高搜索效率

#### Acceptance Criteria

1. WHEN 请求策略推荐时，THE Knowledge Base SHALL 根据op_type和hardware查询动态性能数据
2. THE Knowledge Base SHALL 计算每个策略的质量分数Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)
3. THE Knowledge Base SHALL 返回质量分数最高的top-k个策略及其统计信息
4. THE Strategy System SHALL 要求最少样本数（默认3个）才将策略纳入推荐候选
5. WHEN 没有足够历史数据时，THE Strategy System SHALL 根据op_types字段返回适用的基础策略列表

### Requirement 5

**User Story:** 作为系统集成者，我需要将新的策略定义格式与现有的搜索编排器和生成链集成，以便保持系统功能完整性

#### Acceptance Criteria

1. THE Search Orchestrator SHALL 使用Strategy Registry加载策略定义并获取prompt_hint
2. THE Generation Chain SHALL 接收包含triton_preferences和constraints的策略配置
3. WHEN 生成kernel时，THE Generation Chain SHALL 将策略的prompt_hint、triton_preferences和constraints注入到LLM提示中
4. THE Search Orchestrator SHALL 在每次迭代后将性能结果记录到Knowledge Base的动态性能数据中
5. THE Strategy System SHALL 保持与现有OperatorKey、HardwareSignature和StrategyStats类型的兼容性

### Requirement 6

**User Story:** 作为数据分析师，我希望能够查看和分析特定算子在特定硬件上的策略性能，以便优化策略选择

#### Acceptance Criteria

1. THE Knowledge Base SHALL 提供按op_type和hardware过滤性能数据的查询接口
2. THE Knowledge Base SHALL 支持导出性能数据为人类可读的JSON格式
3. THE Dynamic Performance Data SHALL 包含每个策略的平均加速比、标准差、成功率和覆盖的形状桶信息
4. THE Strategy System SHALL 提供命令行工具查看特定算子-硬件组合的所有策略排名
5. THE Knowledge Base SHALL 支持生成性能对比报告，显示不同策略在相同条件下的表现差异

### Requirement 7

**User Story:** 作为系统维护者，我需要支持策略定义的版本管理，以便跟踪策略演化和回滚变更

#### Acceptance Criteria

1. THE Static Strategy Definition SHALL 包含version字段标识策略版本（如"v1"、"v2"）
2. THE Strategy Registry SHALL 支持同一优化方法的多个版本共存（如"matmul.tile_vec.v1"和"matmul.tile_vec.v2"）
3. WHEN 加载策略定义时，THE Strategy System SHALL 验证version字段格式的有效性
4. THE Dynamic Performance Data SHALL 独立跟踪每个版本的性能统计
5. THE Strategy System SHALL 在metadata中记录策略的变更历史和废弃信息

### Requirement 8

**User Story:** 作为开发者，我希望能够为特定算子类型定义专用策略，以便提高针对性优化的效果

#### Acceptance Criteria

1. THE Static Strategy Definition SHALL 支持op_types字段指定适用的算子类型列表（如["matmul", "batch_matmul"]）
2. WHEN op_types为空列表时，THE Strategy System SHALL 将该策略视为通用策略，适用于所有算子类型
3. THE Strategy Registry SHALL 提供查询接口，返回适用于特定op_type的所有策略
4. THE Operation Service SHALL 使用LLM识别算子类型，结果用于过滤适用策略
5. THE Strategy System SHALL 在推荐策略时优先考虑op_types匹配的专用策略

### Requirement 9

**User Story:** 作为系统，我需要支持策略定义的热重载，以便在不中断运行的情况下更新策略配置

#### Acceptance Criteria

1. THE Strategy Registry SHALL 提供reload()方法重新加载策略定义文件
2. WHEN 策略定义文件被修改时，THE Strategy System SHALL 检测文件变更并触发重载
3. THE Strategy Registry SHALL 在重载时保留现有的动态性能数据
4. WHEN 重载失败时，THE Strategy System SHALL 保持使用旧的策略定义并记录错误日志
5. THE Strategy System SHALL 提供API端点或命令行命令手动触发策略重载

### Requirement 10

**User Story:** 作为开发者，我希望能够定义策略的约束条件，以便限制策略的适用范围和参数空间

#### Acceptance Criteria

1. THE Static Strategy Definition SHALL 支持constraints字段定义形状、数据类型和硬件约束
2. THE constraints字段 SHALL 支持min_shape、max_shape、supported_dtypes和required_hardware_features子字段
3. WHEN 选择策略时，THE Strategy System SHALL 验证当前算子和硬件是否满足策略约束
4. THE Strategy System SHALL 过滤掉不满足约束条件的策略，不将其纳入推荐候选
5. THE constraints字段 SHALL 支持可选配置，未指定时表示无约束限制
