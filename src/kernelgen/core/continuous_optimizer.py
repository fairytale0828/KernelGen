"""
持续优化器
固定策略的持续改进模式，通过错误反馈和策略融合实现优化
"""

import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .types import (
    OperatorKey, HardwareSignature, StrategyId, WorkerId,
    WorkerResult, SearchState
)
from .strategy_registry import StrategyRegistry, get_global_registry
from .knowledge_base import KnowledgeBase
from .experience_pool import ExperiencePool, ExperienceEntry

logger = logging.getLogger(__name__)


@dataclass
class ContinuousWorker:
    """持续优化的Worker"""
    worker_id: WorkerId
    strategy_id: StrategyId
    current_code: Optional[str] = None
    best_sigma: float = 0.0
    best_code: Optional[str] = None
    failure_count: int = 0
    success_count: int = 0
    last_error: str = ""
    is_active: bool = True
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)


class ContinuousOptimizer:
    """
    持续优化器
    
    与SearchOrchestrator不同，这个优化器：
    1. 固定每个worker的策略
    2. 通过错误反馈持续改进
    3. 动态管理worker（融合/淘汰）
    4. 不是每轮换策略，而是持续优化同一策略
    """
    
    def __init__(self,
                 analysis_chain,
                 generation_chain,
                 validation_chain,
                 config,
                 knowledge_base: Optional[KnowledgeBase] = None,
                 experience_pool: Optional[ExperiencePool] = None,
                 strategy_registry: Optional[StrategyRegistry] = None):
        """初始化持续优化器"""
        self.analysis_chain = analysis_chain
        self.generation_chain = generation_chain
        self.validation_chain = validation_chain
        self.config = config
        
        self.kb = knowledge_base or KnowledgeBase()
        self.exp_pool = experience_pool or ExperiencePool(max_size=100)
        self.registry = strategy_registry or get_global_registry()
        
        # Worker管理
        self.workers: Dict[WorkerId, ContinuousWorker] = {}
        self.fusion_counter = 0
        
        # 全局最佳
        self.global_best_sigma = 0.0
        self.global_best_code = None
        self.global_best_strategy = None
        
        logger.info(f"初始化持续优化器: {config.num_workers} workers, {config.max_rounds} rounds")
    
    def _initialize_workers(self, num_workers: int) -> None:
        """初始化固定策略的workers"""
        base_strategies = self.registry.get_base_strategies()
        
        # 随机选择初始策略
        if len(base_strategies) >= num_workers:
            selected = random.sample(base_strategies, num_workers)
        else:
            selected = random.choices(base_strategies, k=num_workers)
        
        for i, strategy_id in enumerate(selected):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = ContinuousWorker(
                worker_id=worker_id,
                strategy_id=strategy_id
            )
        
        logger.info(f"初始化 {len(self.workers)} 个workers:")
        for worker in self.workers.values():
            logger.info(f"  {worker.worker_id}: {worker.strategy_id}")
    
    async def optimize(self,
                      problem_info: Dict[str, Any],
                      pytorch_code: str,
                      pytorch_forward,
                      test_inputs,
                      init_inputs) -> Dict[str, Any]:
        """
        持续优化主循环
        
        与search_best_kernel不同：
        - 固定worker策略
        - 通过错误反馈改进
        - 动态融合和淘汰
        """
        logger.info("=" * 80)
        logger.info("开始持续优化模式")
        logger.info("=" * 80)
        
        # 创建标识
        from .search_orchestrator import SearchOrchestrator
        orchestrator = SearchOrchestrator(
            self.analysis_chain,
            self.generation_chain,
            self.validation_chain,
            self.config,
            self.kb,
            self.exp_pool,
            self.registry
        )
        
        op_key = orchestrator._create_operator_key(problem_info)
        hw = orchestrator._create_hardware_signature()
        
        logger.info(f"算子: {op_key}")
        logger.info(f"硬件: {hw}")
        
        # 运行分析（一次性）
        logger.info("\n🔍 运行分析阶段...")
        analysis_result = await self.analysis_chain.analyze_operation(
            pytorch_code=pytorch_code,
            problem_info=problem_info,
            iteration=1
        )
        
        if not analysis_result.get("success", False):
            return {"success": False, "error": "分析失败"}
        
        analysis_output = analysis_result["result"]
        logger.info("✅ 分析阶段完成")
        
        # 初始化workers
        self._initialize_workers(self.config.num_workers)
        
        # 持续优化循环
        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"🔄 轮次 {round_num}/{self.config.max_rounds}")
            logger.info(f"活跃workers: {sum(1 for w in self.workers.values() if w.is_active)}")
            logger.info(f"{'=' * 80}")
            
            # 执行所有活跃workers
            await self._execute_round_continuous(
                round_num, op_key, hw, analysis_output,
                pytorch_code, problem_info,
                pytorch_forward, test_inputs, init_inputs
            )
            
            # 检查融合机会
            self._check_fusion_opportunities()
            
            # 淘汰失败的workers
            self._eliminate_failed_workers()
            
            # 打印摘要
            self._print_round_summary(round_num)
            
            # 检查是否所有workers都失败
            if not any(w.is_active for w in self.workers.values()):
                logger.warning("所有workers都已失败，提前结束")
                break
        
        # 生成最终结果
        return self._generate_final_result(op_key, hw, problem_info)
    
    async def _execute_round_continuous(self,
                                       round_num: int,
                                       op_key: OperatorKey,
                                       hw: HardwareSignature,
                                       analysis_output: Dict[str, Any],
                                       pytorch_code: str,
                                       problem_info: Dict[str, Any],
                                       pytorch_forward,
                                       test_inputs,
                                       init_inputs) -> None:
        """执行一轮持续优化"""
        
        # 只执行活跃的workers
        active_workers = [w for w in self.workers.values() if w.is_active]
        
        if not active_workers:
            return
        
        logger.info(f"执行 {len(active_workers)} 个活跃workers...")
        
        # 创建任务
        tasks = []
        for worker in active_workers:
            task = self._run_worker_continuous(
                worker, round_num, op_key, hw,
                analysis_output, pytorch_code, problem_info,
                pytorch_forward, test_inputs, init_inputs
            )
            tasks.append(task)
        
        # 并行执行
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_worker_continuous(self,
                                    worker: ContinuousWorker,
                                    round_num: int,
                                    op_key: OperatorKey,
                                    hw: HardwareSignature,
                                    analysis_output: Dict[str, Any],
                                    pytorch_code: str,
                                    problem_info: Dict[str, Any],
                                    pytorch_forward,
                                    test_inputs,
                                    init_inputs) -> None:
        """运行单个worker的持续优化"""
        
        logger.info(f"Worker {worker.worker_id} ({worker.strategy_id}) 开始...")
        
        try:
            # 获取策略提示
            strategy_hint = self.registry.get_hint(worker.strategy_id)
            
            # 如果是首次或之前失败，生成新代码
            if worker.current_code is None or worker.last_error:
                # 生成代码
                generation_result = await self.generation_chain.generate_code(
                    pytorch_code=pytorch_code,
                    problem_info=problem_info,
                    architecture_design=analysis_output.get("kernel_structure", {}),
                    implementation_guidance=analysis_output.get("implementation_guidance", {}),
                    strategy_hint=strategy_hint
                )
                
                if not generation_result.get("success"):
                    worker.last_error = generation_result.get("error", "生成失败")
                    worker.failure_count += 1
                    logger.error(f"Worker {worker.worker_id} 生成失败")
                    return
                
                worker.current_code = generation_result["result"].get("kernel_code", "")
            
            # 验证和测试
            from .search_orchestrator import SearchOrchestrator
            orchestrator = SearchOrchestrator(
                self.analysis_chain,
                self.generation_chain,
                self.validation_chain,
                self.config
            )
            
            perf_result = await orchestrator._run_performance_test(
                worker.current_code,
                pytorch_forward,
                test_inputs,
                init_inputs
            )
            
            valid = perf_result.get("correctness", False)
            T_torch = perf_result.get("pytorch_time", 0.0)
            T_kernel = perf_result.get("triton_time", 0.0)
            
            if valid and T_kernel > 0:
                sigma = T_torch / T_kernel
                
                # 更新worker状态
                worker.success_count += 1
                worker.last_error = ""
                
                if sigma > worker.best_sigma:
                    worker.best_sigma = sigma
                    worker.best_code = worker.current_code
                    logger.info(f"✨ Worker {worker.worker_id} 新最佳: {sigma:.3f}x")
                
                # 更新全局最佳
                if sigma > self.global_best_sigma:
                    self.global_best_sigma = sigma
                    self.global_best_code = worker.current_code
                    self.global_best_strategy = worker.strategy_id
                    logger.info(f"🏆 全局新最佳: {sigma:.3f}x ({worker.strategy_id})")
                
                # 记录到知识库
                self.kb.record_observation(
                    op_key, hw, worker.strategy_id,
                    sigma, True, op_key.shape_bucket
                )
                
                # 检查是否加入经验池
                if sigma >= self.config.UB:
                    entry = ExperienceEntry(
                        op_key=op_key,
                        hw=hw,
                        strategy_id=worker.strategy_id,
                        best_kernel_repr=worker.current_code,
                        best_sigma=sigma,
                        UB=self.config.UB,
                        LB=self.config.LB,
                        DT=self.config.DT
                    )
                    self.exp_pool.add(entry)
                    logger.info(f"📝 Worker {worker.worker_id} 加入经验池")
            
            else:
                # 失败
                worker.failure_count += 1
                worker.last_error = perf_result.get("error", "验证失败")
                logger.warning(f"❌ Worker {worker.worker_id} 失败: {worker.last_error[:100]}")
                
                # 记录失败到知识库
                self.kb.record_observation(
                    op_key, hw, worker.strategy_id,
                    0.0, False, op_key.shape_bucket
                )
            
            # 记录历史
            worker.iteration_history.append({
                "round": round_num,
                "sigma": sigma if valid else 0.0,
                "valid": valid,
                "error": worker.last_error
            })
            
        except Exception as e:
            worker.failure_count += 1
            worker.last_error = str(e)
            logger.error(f"Worker {worker.worker_id} 异常: {e}")
    
    def _check_fusion_opportunities(self) -> None:
        """检查并执行策略融合"""
        
        if self.exp_pool.size() < 2:
            return
        
        # 尝试融合
        fusion_result = self.exp_pool.maybe_spawn_fused_strategy(
            fusion_counter=self.fusion_counter,
            use_best=True
        )
        
        if fusion_result:
            new_strategy_id, fusion_hint, UB_new, LB_new, DT_new = fusion_result
            
            # 注册融合策略
            self.registry.register_fused_strategy(new_strategy_id, fusion_hint)
            
            # 创建新worker替换表现最差的worker
            worst_worker = min(
                [w for w in self.workers.values() if w.is_active],
                key=lambda w: w.best_sigma
            )
            
            # 替换
            worst_worker.strategy_id = new_strategy_id
            worst_worker.current_code = None
            worst_worker.best_sigma = 0.0
            worst_worker.failure_count = 0
            worst_worker.last_error = ""
            
            self.fusion_counter += 1
            
            logger.info(
                f"🔬 融合策略 {new_strategy_id} 替换 {worst_worker.worker_id}"
            )
    
    def _eliminate_failed_workers(self, max_failures: int = 3) -> None:
        """淘汰连续失败的workers"""
        
        for worker in self.workers.values():
            if worker.is_active and worker.failure_count >= max_failures:
                worker.is_active = False
                logger.warning(
                    f"⚠️  Worker {worker.worker_id} ({worker.strategy_id}) "
                    f"连续失败{worker.failure_count}次，已淘汰"
                )
    
    def _print_round_summary(self, round_num: int) -> None:
        """打印轮次摘要"""
        active_count = sum(1 for w in self.workers.values() if w.is_active)
        successful = [w for w in self.workers.values() if w.best_sigma > 0]
        
        logger.info(f"\n📊 轮次 {round_num} 摘要:")
        logger.info(f"  活跃workers: {active_count}/{len(self.workers)}")
        logger.info(f"  成功workers: {len(successful)}")
        
        if self.global_best_sigma > 0:
            logger.info(f"  全局最佳: {self.global_best_strategy} - {self.global_best_sigma:.3f}x")
        
        logger.info(f"  经验池大小: {self.exp_pool.size()}")
    
    def _generate_final_result(self,
                              op_key: OperatorKey,
                              hw: HardwareSignature,
                              problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终结果"""
        
        return {
            "success": self.global_best_sigma > 0,
            "best_speedup": self.global_best_sigma,
            "best_kernel": self.global_best_code,
            "best_strategy": self.global_best_strategy,
            "total_workers": len(self.workers),
            "active_workers": sum(1 for w in self.workers.values() if w.is_active),
            "experience_pool_size": self.exp_pool.size(),
            "knowledge_base_size": self.kb.size()
        }
