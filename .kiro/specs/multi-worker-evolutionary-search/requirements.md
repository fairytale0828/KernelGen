# 需求文档

## 介绍

本文档规定了将现有单worker三Agent系统（analysis_chain、generation_chain、validation_chain）重构为多worker进化式搜索架构的需求。系统将保持现有链功能的同时，增加并行worker执行、基于策略的优化、经验池和硬件-算子知识库功能。

## 术语表

- **Analysis_Chain**: 现有的LangChain组件，分析PyTorch代码和硬件信息以产生结构化分析结果
- **Generation_Chain**: 现有的LangChain组件，基于分析结果和策略提示生成Triton kernel代码
- **Validation_Chain**: 现有的LangChain组件，验证生成的kernel的正确性和性能
- **Search_Orchestrator**: 新组件，协调多个worker并管理进化式搜索过程
- **Strategy**: 特定的优化方法（如"Triton.TileOnly.v1"），包含用于代码生成的相关提示文本
- **Worker**: 并行执行单元，为特定策略运行一次analysis->generation->validation迭代
- **Experience_Pool**: 高质量kernel实现的存储系统，可用于策略融合
- **Knowledge_Base**: 长期存储系统，跟踪不同算子-硬件组合下的策略性能
- **OperatorKey**: 操作类型的唯一标识符，包括操作名称、数据类型和形状桶
- **HardwareSignature**: 硬件配置的唯一标识符，包括设备规格和功能
- **Speedup**: 性能比率，计算为T_kernel / T_torch，其中T_torch是PyTorch执行时间，T_kernel是Triton kernel执行时间

## 需求

### 需求 1

**用户故事:** 作为kernel优化研究员，我希望系统能够并行运行多种优化策略，以便我能同时探索不同方法并更快找到更好的解决方案。

#### 验收标准

1. WHEN 系统开始kernel优化时，THE Search_Orchestrator SHALL 生成N_workers个并行worker，其中N_workers是可配置的
2. WHEN worker被生成时，THE Search_Orchestrator SHALL 从可用策略注册表中为每个worker分配不同的策略
3. WHILE worker正在执行时，THE Search_Orchestrator SHALL 协调它们的执行而不相互阻塞
4. WHEN 所有worker完成任务时，THE Search_Orchestrator SHALL 收集并比较它们的结果
5. THE Search_Orchestrator SHALL 维护每个策略的性能记录，包括最佳加速比和最佳kernel代码

### 需求 2

**用户故事:** 作为系统架构师，我希望保留现有的三链架构，以便当前的分析、生成和验证逻辑保持完整和功能正常。

#### 验收标准

1. THE Analysis_Chain SHALL 继续接受PyTorch代码和硬件信息作为输入
2. THE Generation_Chain SHALL 增强以接受额外的strategy_hint参数，同时保持向后兼容性
3. THE Validation_Chain SHALL 继续验证kernel并返回正确性和性能指标
4. WHEN Search_Orchestrator调用链时，THE 链 SHALL 执行其现有逻辑而不修改核心功能
5. THE 系统 SHALL 维护所有现有服务依赖，包括HardwareInfoService、KnowledgeBaseService等

### 需求 3

**用户故事:** 作为性能优化专家，我希望系统能从成功的优化中学习，以便将经过验证的策略应用到未来的类似问题中。

#### 验收标准

1. THE Knowledge_Base SHALL 存储按OperatorKey、HardwareSignature和StrategyId索引的策略性能统计
2. WHEN worker完成执行时，THE Knowledge_Base SHALL 记录加速比、成功状态和形状桶信息
3. THE Knowledge_Base SHALL 使用公式计算质量分数：Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)
4. WHEN 为新迭代选择策略时，THE Search_Orchestrator SHALL 查询Knowledge_Base以获取表现最佳的策略
5. THE Knowledge_Base SHALL 支持持久化到磁盘，以便跨会话进行长期学习

### 需求 4

**用户故事:** 作为kernel开发者，我希望系统能自动融合成功的策略，以便通过组合经过验证的技术来发现新的优化方法。

#### 验收标准

1. WHEN worker达到加速比 >= UB阈值时，THE Experience_Pool SHALL 存储成功的kernel和策略信息
2. WHEN Experience_Pool包含 >= 2个条目时，THE 系统 SHALL 支持融合两个经验以创建新策略
3. THE 融合策略 SHALL 具有新的StrategyId和来自两个父策略的组合优化提示
4. THE 融合策略 SHALL 继承按以下方式计算的阈值：UB_new = max(UB_a, UB_b)，LB_new = (LB_a + LB_b)/2，DT_new = (DT_a + DT_b)/2
5. THE Search_Orchestrator SHALL 在后续worker分配中包含融合策略

### 需求 5

**用户故事:** 作为系统操作员，我希望系统能检测性能退化并触发干预，以便优化工作保持专注和高效。

#### 验收标准

1. THE Search_Orchestrator SHALL 为每个策略计算退化量：Δ_s^(t) = σ_s^* - σ_s^(t)
2. WHEN 1 < σ_s^(t) < LB 或 Δ_s^(t) > DT时，THE 系统 SHALL 标记该策略需要人工干预
3. WHEN σ_s^(t) >= UB时，THE 系统 SHALL 将结果添加到Experience_Pool
4. THE 系统 SHALL 为每个优化会话维护可配置的阈值UB、LB和DT
5. THE Search_Orchestrator SHALL 记录所有阈值违规和干预触发

### 需求 6

**用户故事:** 作为研究员，我希望系统实现预热阶段，然后进行智能策略选择，以便有效平衡探索和利用。

#### 验收标准

1. WHEN 开始优化时，THE Search_Orchestrator SHALL 使用随机选择的策略执行预热轮次
2. WHEN 预热完成时，THE Search_Orchestrator SHALL 运行一次Analysis_Chain并在所有worker间共享结果
3. WHEN 为后续轮次选择策略时，THE Search_Orchestrator SHALL 基于当前best_sigma值选择前N/2个策略
4. THE Search_Orchestrator SHALL 从可用策略池中随机采样剩余的N/2个策略
5. THE Search_Orchestrator SHALL 在当前OperatorKey和HardwareSignature可用时纳入Knowledge_Base推荐

### 需求 7

**用户故事:** 作为性能工程师，我希望系统跟踪每个策略的综合统计信息，以便了解哪些方法在不同场景下效果最好。

#### 验收标准

1. THE StrategyStats SHALL 维护运行统计，包括样本数、加速比总和、加速比平方和以及成功计数
2. THE StrategyStats SHALL 跟踪遇到的唯一形状桶以衡量策略覆盖范围
3. THE StrategyStats SHALL 计算平均加速比：μ_s = (1/|D_s|)Σσ_i 和方差：v_s = (1/|D_s|)Σ(σ_i - μ_s)²
4. THE StrategyStats SHALL 计算成功率：p_s^succ = (成功实验数) / |D_s|
5. THE StrategyStats SHALL 提供结合所有指标的质量分数，使用可配置权重α、β、γ

### 需求 8

**用户故事:** 作为系统集成员，我希望新架构能与现有配置和日志系统无缝集成，以便部署和监控保持一致。

#### 验收标准

1. THE Search_Orchestrator SHALL 从现有config.yaml文件读取配置，包括worker数量和阈值
2. THE 系统 SHALL 与现有IterationLogger集成以跟踪多worker执行
3. THE Search_Orchestrator SHALL 保持与现有性能基准测试工具的兼容性
4. THE 系统 SHALL 保留所有现有输出格式和文件结构
5. THE Search_Orchestrator SHALL 支持与当前系统相同的LLM后端和API配置