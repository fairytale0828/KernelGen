# 实现计划

- [ ] 1. 创建核心类型和数据结构
  - 实现OperatorKey、HardwareSignature等基础类型定义
  - 创建WorkerResult、SearchState等执行状态模型
  - 添加必要的类型注解和文档字符串
  - _需求: 1.1, 7.1, 7.2_

- [x] 1.1 实现核心类型定义模块
  - 在src/kernelgen/core/types.py中创建OperatorKey和HardwareSignature数据类
  - 定义StrategyId和WorkerId类型别名
  - 添加数据验证和序列化支持
  - _需求: 1.1, 2.1_

- [x] 1.2 创建搜索状态和结果模型
  - 实现SearchState类用于跟踪搜索进度
  - 实现WorkerResult类用于封装worker执行结果
  - 添加结果聚合和比较方法
  - _需求: 1.4, 1.5_

- [ ] 2. 实现策略统计和质量评分系统
  - 创建StrategyStats类实现长期性能统计
  - 实现质量评分公式Q_s = μ_s - α√v_s + βp_s^succ + γlog(1 + c_s)
  - 添加样本更新和统计计算方法
  - _需求: 3.3, 7.3, 7.4, 7.5_

- [x] 2.1 实现StrategyStats统计类
  - 在src/kernelgen/core/strategy_stats.py中实现统计数据结构
  - 实现add_sample方法更新统计量
  - 实现mu、var、p_success、coverage属性计算
  - _需求: 7.1, 7.2, 7.3_

- [x] 2.2 实现质量评分计算
  - 实现quality属性按照设计公式计算质量分数
  - 添加可配置的权重参数α、β、γ
  - 实现统计数据的序列化和反序列化
  - _需求: 3.3, 7.5_

- [ ] 3. 创建策略注册表和提示系统
  - 实现策略注册表维护所有可用策略
  - 创建策略提示文本映射
  - 支持动态策略注册和融合策略管理
  - _需求: 1.2, 4.3, 4.4_

- [x] 3.1 实现策略注册表
  - 在src/kernelgen/core/strategy_registry.py中定义ALL_STRATEGIES列表
  - 创建STRATEGY_HINTS字典映射策略ID到提示文本
  - 实现策略验证和查询方法
  - _需求: 1.2, 2.2_

- [x] 3.2 扩展Generation_Chain支持策略提示
  - 修改generation_chain的generate_code方法接受strategy_hint参数
  - 更新prompt模板添加{strategy_hint}占位符
  - 确保向后兼容性，strategy_hint为可选参数
  - _需求: 2.2, 2.4_

- [ ] 4. 实现硬件-算子知识库
  - 创建KnowledgeBase类管理策略性能历史
  - 实现按(OperatorKey, HardwareSignature, StrategyId)索引的存储
  - 添加top策略查询和持久化功能
  - _需求: 3.1, 3.2, 3.4, 3.5_

- [x] 4.1 实现KnowledgeBase核心功能
  - 在src/kernelgen/core/knowledge_base.py中创建知识库类
  - 实现record_observation方法记录策略执行结果
  - 实现get_top_strategies方法按质量分数排序返回最佳策略
  - _需求: 3.1, 3.2, 3.4_

- [x] 4.2 添加知识库持久化支持
  - 实现save和load方法支持磁盘持久化
  - 添加数据版本控制和迁移支持
  - 实现增量更新和备份机制
  - _需求: 3.5_

- [ ] 5. 实现经验池和策略融合
  - 创建ExperiencePool管理高质量kernel经验
  - 实现策略融合逻辑生成新的优化策略
  - 添加融合策略的提示文本生成
  - _需求: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5.1 实现ExperienceEntry和ExperiencePool
  - 在src/kernelgen/core/experience_pool.py中创建经验条目数据结构
  - 实现经验池的添加、查询和采样方法
  - 添加经验池大小管理和清理机制
  - _需求: 4.1, 4.2_

- [x] 5.2 实现策略融合逻辑
  - 实现fuse_to_new_strategy方法融合两个经验
  - 生成新的StrategyId和组合优化提示
  - 计算融合策略的阈值参数
  - _需求: 4.3, 4.4_

- [x] 5.3 集成融合策略到注册表
  - 扩展策略注册表支持动态添加融合策略
  - 实现融合策略提示文本的智能生成
  - 添加融合策略的生命周期管理
  - _需求: 4.5_

- [ ] 6. 实现Search Orchestrator核心协调逻辑
  - 创建SearchOrchestrator类协调多worker执行
  - 实现worker调度和结果收集机制
  - 添加per-strategy最优记录维护
  - _需求: 1.1, 1.3, 1.4, 1.5_

- [x] 6.1 创建SearchOrchestrator基础框架
  - 在src/kernelgen/core/search_orchestrator.py中创建协调器类
  - 初始化知识库、经验池和策略注册表依赖
  - 实现基础的配置管理和状态跟踪
  - _需求: 1.1, 8.1_

- [x] 6.2 实现单worker执行逻辑
  - 实现run_worker_once方法调用三条chain
  - 计算speedup和reward按照公式σ = T_torch / T_kernel
  - 更新per-strategy最优记录best_sigma和best_kernel
  - _需求: 1.4, 1.5_

- [x] 6.3 实现worker结果处理和知识库更新
  - 处理worker执行结果并更新统计信息
  - 调用Knowledge_Base.record_observation记录观察结果
  - 实现错误处理和异常情况管理
  - _需求: 3.2, 1.5_

- [ ] 7. 实现搜索主循环和策略选择
  - 实现warmup阶段的随机策略选择
  - 实现后续轮次的Top-K + 随机策略选择
  - 添加阈值判断和经验池写入逻辑
  - _需求: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 7.1 实现warmup阶段逻辑
  - 实现search_best_kernel方法的warmup轮次
  - 运行一次Analysis_Chain并在所有worker间共享结果
  - 随机选择N_workers个策略并分配给worker
  - _需求: 6.1, 6.2_

- [x] 7.2 实现智能策略选择算法
  - 实现Top N/2策略选择基于当前best_sigma值
  - 实现剩余N/2策略的随机采样
  - 集成Knowledge_Base推荐到策略选择过程
  - _需求: 6.3, 6.4, 6.5_

- [x] 7.3 实现阈值判断和干预逻辑
  - 计算策略退化量Δ_s^(t) = σ_s^* - σ_s^(t)
  - 实现人工干预条件判断(1 < σ_s^(t) < LB OR Δ_s^(t) > DT)
  - 实现经验池写入条件判断(σ_s^(t) >= UB)
  - _需求: 5.1, 5.2, 5.3, 5.4_

- [ ] 8. 实现并行worker执行和协调
  - 添加异步并行执行支持
  - 实现worker间的资源协调和冲突避免
  - 添加超时和错误恢复机制
  - _需求: 1.1, 1.2, 1.3_

- [x] 8.1 实现异步worker执行
  - 使用asyncio实现多worker并行执行
  - 实现worker任务分发和结果收集
  - 添加worker执行状态监控
  - _需求: 1.1, 1.2, 1.3_

- [x] 8.2 添加资源管理和冲突避免
  - 实现GPU内存使用协调避免OOM
  - 添加worker执行超时和重试机制
  - 实现优雅的错误处理和恢复
  - _需求: 1.3_

- [ ] 9. 扩展配置系统支持多worker参数
  - 扩展config.yaml添加多worker搜索配置
  - 实现配置验证和默认值处理
  - 添加运行时配置更新支持
  - _需求: 8.1, 5.4_

- [x] 9.1 扩展配置文件结构
  - 在config.yaml中添加multi_worker_search配置节
  - 定义worker数量、阈值、策略选择等参数
  - 实现配置加载和验证逻辑
  - _需求: 8.1, 5.4_

- [x] 9.2 实现配置管理类
  - 创建SearchConfig类管理多worker搜索配置
  - 实现配置参数的类型检查和范围验证
  - 添加配置热更新和持久化支持
  - _需求: 8.1_

- [ ] 10. 集成现有系统和向后兼容
  - 集成IterationLogger支持多worker日志记录
  - 保持与现有性能基准测试工具的兼容性
  - 确保现有API接口的向后兼容性
  - _需求: 8.2, 8.3, 8.4, 8.5_

- [x] 10.1 扩展IterationLogger支持多worker
  - 修改IterationLogger支持并行worker执行记录
  - 添加worker级别的性能指标跟踪
  - 实现搜索过程的详细日志记录
  - _需求: 8.2_

- [x] 10.2 确保性能基准测试兼容性
  - 验证SearchOrchestrator与TritonPerformanceBenchmark的集成
  - 保持现有性能测试接口和输出格式
  - 添加多worker性能测试的聚合统计
  - _需求: 8.3, 8.4_

- [x] 10.3 实现向后兼容接口
  - 创建兼容层支持现有OrchestrationChain接口
  - 实现单worker模式作为fallback选项
  - 确保现有LLM后端和API配置正常工作
  - _需求: 2.1, 2.3, 2.4, 2.5, 8.5_

- [ ] 11. 添加全面测试覆盖
  - 创建单元测试覆盖所有核心组件
  - 实现集成测试验证多worker协调
  - 添加性能测试和压力测试
  - _需求: 所有需求的验证_

- [ ] 11.1 实现核心组件单元测试
  - 为StrategyStats、KnowledgeBase、ExperiencePool创建单元测试
  - 测试统计计算、存储检索、融合逻辑的正确性
  - 添加边界条件和异常情况测试
  - _需求: 2.1, 3.1, 4.1_

- [ ] 11.2 实现集成测试套件
  - 测试SearchOrchestrator的完整搜索流程
  - 验证多worker并行执行和结果聚合
  - 测试策略选择和阈值判断逻辑
  - _需求: 1.1, 6.1, 7.1_

- [ ] 11.3 添加性能和压力测试
  - 测试多worker并发执行的性能表现
  - 验证知识库和经验池的内存使用效率
  - 测试搜索算法的收敛性和稳定性
  - _需求: 1.3, 3.5, 4.2_

- [ ] 12. 创建文档和使用示例
  - 编写API文档和架构说明
  - 创建配置指南和最佳实践
  - 提供完整的使用示例和教程
  - _需求: 8.1, 8.4_

- [x] 12.1 编写技术文档
  - 创建多worker搜索架构的详细文档
  - 编写各组件的API参考文档
  - 添加配置参数说明和调优指南
  - _需求: 8.1, 8.4_

- [x] 12.2 创建使用示例和教程
  - 提供从单worker迁移到多worker的指南
  - 创建自定义策略开发教程
  - 添加性能调优和故障排除指南
  - _需求: 8.4, 8.5_