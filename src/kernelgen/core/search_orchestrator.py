"""
搜索协调器模块
协调多worker并行执行和进化式搜索
"""

import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .types import (
    OperatorKey, HardwareSignature, StrategyId, WorkerId,
    WorkerResult, SearchState
)
from .strategy_registry import StrategyRegistry, get_global_registry
from .strategy_stats import StrategyStats
from .knowledge_base import KnowledgeBase
from .experience_pool import ExperiencePool, ExperienceEntry

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """
    搜索配置
    
    Attributes:
        num_workers: 并行worker数量
        max_rounds: 最大搜索轮次
        UB: 上界阈值（写入经验池）
        LB: 下界阈值（人工干预）
        DT: 退化阈值（人工干预）
        top_k_ratio: Top-K策略比例
        knowledge_base_weight: 知识库推荐权重
        fusion_probability: 融合概率
        min_samples_for_recommendation: 知识库推荐最小样本数
    """
    num_workers: int = 4
    max_rounds: int = 10
    UB: float = 2.0
    LB: float = 1.2
    DT: float = 0.5
    top_k_ratio: float = 0.5
    knowledge_base_weight: float = 0.3
    fusion_probability: float = 0.1
    min_samples_for_recommendation: int = 3
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SearchConfig":
        """从配置字典创建实例"""
        return cls(
            num_workers=config_dict.get("num_workers", 4),
            max_rounds=config_dict.get("max_rounds", 10),
            UB=config_dict.get("thresholds", {}).get("UB", 2.0),
            LB=config_dict.get("thresholds", {}).get("LB", 1.2),
            DT=config_dict.get("thresholds", {}).get("DT", 0.5),
            top_k_ratio=config_dict.get("strategy_selection", {}).get("top_k_ratio", 0.5),
            knowledge_base_weight=config_dict.get("strategy_selection", {}).get("knowledge_base_weight", 0.3),
            fusion_probability=config_dict.get("experience_pool", {}).get("fusion_probability", 0.1),
            min_samples_for_recommendation=config_dict.get("knowledge_base", {}).get("min_samples_for_recommendation", 3)
        )


class SearchOrchestrator:
    """
    搜索协调器
    
    协调多个worker并行执行，实现进化式搜索策略。
    """
    
    def __init__(self,
                 analysis_chain,
                 generation_chain,
                 validation_chain,
                 config: SearchConfig,
                 knowledge_base: Optional[KnowledgeBase] = None,
                 experience_pool: Optional[ExperiencePool] = None,
                 strategy_registry: Optional[StrategyRegistry] = None):
        """
        初始化搜索协调器
        
        Args:
            analysis_chain: 分析链
            generation_chain: 生成链
            validation_chain: 验证链
            config: 搜索配置
            knowledge_base: 知识库（可选）
            experience_pool: 经验池（可选）
            strategy_registry: 策略注册表（可选）
        """
        self.analysis_chain = analysis_chain
        self.generation_chain = generation_chain
        self.validation_chain = validation_chain
        self.config = config
        
        # 初始化数据结构
        self.kb = knowledge_base or KnowledgeBase()
        self.exp_pool = experience_pool or ExperiencePool(max_size=100)
        self.registry = strategy_registry or get_global_registry()
        
        # 搜索状态
        self.state = SearchState()
        
        # 融合计数器
        self._fusion_counter = 0
        
        logger.info(f"初始化搜索协调器: {config.num_workers} workers, {config.max_rounds} rounds")
    
    def _create_operator_key(self, problem_info: Dict[str, Any]) -> OperatorKey:
        """
        从问题信息创建算子标识符
        
        Args:
            problem_info: 问题信息
            
        Returns:
            OperatorKey实例
        """
        # 提取操作类型
        op_type = problem_info.get("operation_name", "unknown")
        
        # 提取数据类型
        dtypes = tuple(problem_info.get("dtypes", ["fp32"]))
        
        # 创建形状桶（简化版本，将形状归类到桶中）
        input_shapes = problem_info.get("input_shapes", [])
        if input_shapes:
            # 使用第一个输入的形状作为桶
            first_shape = input_shapes[0] if isinstance(input_shapes[0], (list, tuple)) else [input_shapes[0]]
            # 将形状归类到2的幂次桶中
            shape_bucket = tuple(2 ** (len(bin(max(1, s))) - 2) for s in first_shape)
        else:
            shape_bucket = (1024,)  # 默认桶
        
        return OperatorKey(op_type=op_type, dtypes=dtypes, shape_bucket=shape_bucket)
    
    def _create_hardware_signature(self) -> HardwareSignature:
        """
        创建硬件签名
        
        Returns:
            HardwareSignature实例
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                
                # 获取内存带宽（某些PyTorch版本可能没有这个属性）
                try:
                    mem_bandwidth = device_props.memory_bandwidth / 1e9
                except AttributeError:
                    # 估算内存带宽（基于设备类型）
                    if "A100" in device_props.name:
                        mem_bandwidth = 1555.0  # A100: ~1555 GB/s
                    elif "V100" in device_props.name:
                        mem_bandwidth = 900.0   # V100: ~900 GB/s
                    elif "A10" in device_props.name:
                        mem_bandwidth = 600.0   # A10: ~600 GB/s
                    elif "T4" in device_props.name:
                        mem_bandwidth = 300.0   # T4: ~300 GB/s
                    else:
                        mem_bandwidth = 500.0   # 默认估算值
                    logger.info(f"使用估算的内存带宽: {mem_bandwidth} GB/s for {device_props.name}")
                
                return HardwareSignature(
                    device_name=device_props.name,
                    sm_count=device_props.multi_processor_count,
                    shared_mem_per_sm=device_props.shared_memory_per_multiprocessor,
                    regs_per_sm=device_props.regs_per_multiprocessor,
                    mem_bandwidth_gbps=mem_bandwidth,
                    tensor_core_support=device_props.major >= 7  # Volta及以上
                )
            else:
                # CPU fallback
                return HardwareSignature(
                    device_name="CPU",
                    sm_count=1,
                    shared_mem_per_sm=65536,  # 64KB默认值
                    regs_per_sm=65536,
                    mem_bandwidth_gbps=10.0,
                    tensor_core_support=False
                )
        except Exception as e:
            logger.warning(f"无法获取硬件信息: {e}")
            # 返回合理的默认值
            return HardwareSignature(
                device_name="Unknown GPU",
                sm_count=80,
                shared_mem_per_sm=65536,  # 64KB
                regs_per_sm=65536,
                mem_bandwidth_gbps=500.0,
                tensor_core_support=True
            )
    
    def _select_strategies_for_round(self,
                                    round_num: int,
                                    op_key: OperatorKey,
                                    hw: HardwareSignature) -> List[StrategyId]:
        """
        为当前轮次选择策略
        
        Args:
            round_num: 轮次编号
            op_key: 算子标识符
            hw: 硬件签名
            
        Returns:
            策略ID列表
        """
        num_workers = self.config.num_workers
        
        if round_num == 1:
            # Warmup: 随机选择策略
            all_strategies = self.registry.get_base_strategies()
            if len(all_strategies) >= num_workers:
                selected = random.sample(all_strategies, num_workers)
            else:
                # 如果策略不够，重复选择
                selected = random.choices(all_strategies, k=num_workers)
            
            logger.info(f"Warmup轮次: 随机选择 {len(selected)} 个策略")
            return selected
        
        # 后续轮次: Top-K + 随机 + 知识库推荐
        num_top = int(num_workers * self.config.top_k_ratio)
        num_random = num_workers - num_top
        
        # 1. 选择Top-K策略（基于当前best_sigma）
        top_strategies = self.state.get_top_strategies(num_top)
        
        # 2. 从知识库获取推荐
        kb_recommendations = self.kb.get_top_strategies(
            op_key, hw,
            k=3,
            min_samples=self.config.min_samples_for_recommendation
        )
        kb_strategy_ids = [stats.strategy_id for stats in kb_recommendations]
        
        # 3. 随机选择剩余策略
        all_strategies = self.registry.get_all_strategies()
        # 排除已选择的top策略
        remaining = [s for s in all_strategies if s not in top_strategies]
        
        # 优先考虑知识库推荐
        random_strategies = []
        if kb_strategy_ids and random.random() < self.config.knowledge_base_weight:
            # 从知识库推荐中选择
            available_kb = [s for s in kb_strategy_ids if s in remaining]
            if available_kb:
                random_strategies.append(random.choice(available_kb))
                num_random -= 1
        
        # 填充剩余的随机策略
        if num_random > 0 and remaining:
            if len(remaining) >= num_random:
                random_strategies.extend(random.sample(remaining, num_random))
            else:
                random_strategies.extend(random.choices(remaining, k=num_random))
        
        selected = top_strategies + random_strategies
        
        logger.info(
            f"轮次 {round_num}: 选择 {len(top_strategies)} Top + "
            f"{len(random_strategies)} Random (KB推荐: {len(kb_strategy_ids)})"
        )
        
        return selected
    
    def _check_thresholds(self,
                         result: WorkerResult,
                         strategy_id: StrategyId) -> Dict[str, Any]:
        """
        检查阈值条件
        
        Args:
            result: Worker执行结果
            strategy_id: 策略ID
            
        Returns:
            阈值检查结果字典
        """
        sigma = result.sigma
        best_sigma = self.state.best_sigma.get(strategy_id, 0.0)
        
        # 计算退化量
        delta = best_sigma - sigma if best_sigma > 0 else 0.0
        
        # 检查各种条件
        checks = {
            "sigma": sigma,
            "best_sigma": best_sigma,
            "delta": delta,
            "needs_intervention": False,
            "add_to_experience": False,
            "intervention_reason": None
        }
        
        # 人工干预条件
        if result.valid:
            if 1.0 < sigma < self.config.LB:
                checks["needs_intervention"] = True
                checks["intervention_reason"] = f"性能低于下界 (sigma={sigma:.2f} < LB={self.config.LB})"
            elif delta > self.config.DT:
                checks["needs_intervention"] = True
                checks["intervention_reason"] = f"性能退化过大 (delta={delta:.2f} > DT={self.config.DT})"
        
        # 经验池写入条件
        if result.valid and sigma >= self.config.UB:
            checks["add_to_experience"] = True
        
        return checks
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"SearchOrchestrator(workers={self.config.num_workers}, "
                f"round={self.state.round}/{self.config.max_rounds})")
    
    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()


    async def run_worker_once(self,
                             worker_id: WorkerId,
                             strategy_id: StrategyId,
                             op_key: OperatorKey,
                             hw: HardwareSignature,
                             analysis_output: Dict[str, Any],
                             pytorch_code: str,
                             problem_info: Dict[str, Any],
                             pytorch_forward,
                             test_inputs,
                             init_inputs) -> WorkerResult:
        """
        运行单个worker一次
        
        Args:
            worker_id: Worker标识符
            strategy_id: 策略ID
            op_key: 算子标识符
            hw: 硬件签名
            analysis_output: 分析链输出（共享）
            pytorch_code: PyTorch代码
            problem_info: 问题信息
            pytorch_forward: PyTorch forward函数
            test_inputs: 测试输入
            init_inputs: 初始化输入
            
        Returns:
            WorkerResult实例
        """
        logger.info(f"Worker {worker_id} 开始执行策略 {strategy_id}")
        
        try:
            # 1. 获取策略提示
            strategy_hint = self.registry.get_hint(strategy_id)
            if strategy_hint is None:
                logger.warning(f"策略 {strategy_id} 没有提示文本")
                strategy_hint = ""
            
            # 2. 调用generation_chain生成代码
            logger.debug(f"Worker {worker_id}: 调用generation_chain")
            generation_result = await self.generation_chain.generate_code(
                pytorch_code=pytorch_code,
                problem_info=problem_info,
                architecture_design=analysis_output.get("kernel_structure", {}),
                implementation_guidance=analysis_output.get("implementation_guidance", {}),
                strategy_hint=strategy_hint
            )
            
            if not generation_result.get("success", False):
                error_msg = generation_result.get("error", "生成失败")
                logger.error(f"Worker {worker_id}: 生成失败 - {error_msg}")
                return self._create_failed_result(
                    worker_id, strategy_id, op_key, error_msg
                )
            
            kernel_code = generation_result["result"].get("kernel_code", "")
            if not kernel_code:
                logger.error(f"Worker {worker_id}: 生成的代码为空")
                return self._create_failed_result(
                    worker_id, strategy_id, op_key, "生成的代码为空"
                )
            
            # 3. 调用validation_chain进行性能测试
            logger.debug(f"Worker {worker_id}: 调用validation_chain")
            
            # 运行性能测试
            performance_result = await self._run_performance_test(
                kernel_code, pytorch_forward, test_inputs, init_inputs
            )
            
            # 4. 计算speedup和reward
            valid = performance_result.get("correctness", False)
            T_torch = performance_result.get("pytorch_time", 0.0)
            T_kernel = performance_result.get("triton_time", 0.0)
            
            if valid and T_kernel > 0:
                sigma = T_torch / T_kernel
                reward = sigma
            else:
                sigma = 0.0
                reward = -1.0
            
            # 提取形状桶
            shape_bucket = op_key.shape_bucket
            
            # 创建结果
            result = WorkerResult(
                worker_id=worker_id,
                strategy_id=strategy_id,
                kernel_code=kernel_code,
                sigma=sigma,
                reward=reward,
                valid=valid,
                T_torch=T_torch,
                T_kernel=T_kernel,
                shape_bucket=shape_bucket,
                error_info=performance_result.get("error", ""),
                analysis_result=analysis_output,
                generation_result=generation_result,
                validation_result=performance_result
            )
            
            logger.info(
                f"Worker {worker_id} 完成: {strategy_id}, "
                f"valid={valid}, sigma={sigma:.3f}x"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Worker {worker_id} 执行异常: {e}", exc_info=True)
            return self._create_failed_result(
                worker_id, strategy_id, op_key, str(e)
            )
    
    def _create_failed_result(self,
                             worker_id: WorkerId,
                             strategy_id: StrategyId,
                             op_key: OperatorKey,
                             error_msg: str) -> WorkerResult:
        """
        创建失败的WorkerResult
        
        Args:
            worker_id: Worker ID
            strategy_id: 策略ID
            op_key: 算子标识符
            error_msg: 错误信息
            
        Returns:
            失败的WorkerResult
        """
        return WorkerResult(
            worker_id=worker_id,
            strategy_id=strategy_id,
            kernel_code="",
            sigma=0.0,
            reward=-1.0,
            valid=False,
            T_torch=0.0,
            T_kernel=0.0,
            shape_bucket=op_key.shape_bucket,
            error_info=error_msg
        )
    
    async def _run_performance_test(self,
                                   kernel_code: str,
                                   pytorch_forward,
                                   test_inputs,
                                   init_inputs) -> Dict[str, Any]:
        """
        运行性能测试
        
        Args:
            kernel_code: Triton kernel代码
            pytorch_forward: PyTorch forward函数
            test_inputs: 测试输入
            init_inputs: 初始化输入
            
        Returns:
            性能测试结果
        """
        try:
            # 导入性能测试工具
            from ..tools.performance_tools import PerformanceBenchmarkTool
            from .performance_benchmark import TritonPerformanceBenchmark
            
            # 创建性能测试工具
            benchmark = TritonPerformanceBenchmark(
                device="cuda",
                warmup_runs=10,
                benchmark_runs=100
            )
            perf_tool = PerformanceBenchmarkTool(benchmark)
            
            # 运行测试
            result = perf_tool._run_performance_test(
                kernel_code, pytorch_forward, test_inputs, init_inputs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "correctness": False,
                "speedup": 0.0,
                "pytorch_time": 0.0,
                "triton_time": 0.0
            }


    def _process_worker_result(self,
                              result: WorkerResult,
                              op_key: OperatorKey,
                              hw: HardwareSignature) -> Dict[str, Any]:
        """
        处理worker执行结果
        
        Args:
            result: Worker执行结果
            op_key: 算子标识符
            hw: 硬件签名
            
        Returns:
            处理结果摘要
        """
        strategy_id = result.strategy_id
        
        # 1. 更新搜索状态
        self.state.total_evaluations += 1
        if result.valid:
            self.state.successful_evaluations += 1
        
        # 2. 更新per-strategy最优记录
        updated = self.state.update_best(result)
        if updated:
            logger.info(
                f"✨ 新的最佳结果! {strategy_id}: {result.sigma:.3f}x "
                f"(之前: {self.state.best_sigma.get(strategy_id, 0.0):.3f}x)"
            )
        
        # 3. 记录到知识库
        self.kb.record_observation(
            op_key=op_key,
            hw=hw,
            strategy_id=strategy_id,
            sigma=result.sigma,
            success=result.valid,
            shape_bucket=result.shape_bucket
        )
        
        # 4. 检查阈值
        threshold_checks = self._check_thresholds(result, strategy_id)
        
        # 5. 处理经验池写入
        if threshold_checks["add_to_experience"]:
            self._add_to_experience_pool(result, op_key, hw)
        
        # 6. 处理人工干预
        if threshold_checks["needs_intervention"]:
            logger.warning(
                f"⚠️  策略 {strategy_id} 需要人工干预: "
                f"{threshold_checks['intervention_reason']}"
            )
        
        return {
            "updated_best": updated,
            "threshold_checks": threshold_checks,
            "current_best_sigma": self.state.best_sigma.get(strategy_id, 0.0)
        }
    
    def _add_to_experience_pool(self,
                               result: WorkerResult,
                               op_key: OperatorKey,
                               hw: HardwareSignature) -> None:
        """
        将结果添加到经验池
        
        Args:
            result: Worker执行结果
            op_key: 算子标识符
            hw: 硬件签名
        """
        entry = ExperienceEntry(
            op_key=op_key,
            hw=hw,
            strategy_id=result.strategy_id,
            best_kernel_repr=result.kernel_code,
            best_sigma=result.sigma,
            UB=self.config.UB,
            LB=self.config.LB,
            DT=self.config.DT
        )
        
        self.exp_pool.add(entry)
        logger.info(
            f"📝 添加到经验池: {result.strategy_id}, "
            f"sigma={result.sigma:.3f}x (池大小: {self.exp_pool.size()})"
        )
        
        # 尝试生成融合策略
        if random.random() < self.config.fusion_probability:
            self._try_create_fused_strategy()
    
    def _try_create_fused_strategy(self) -> None:
        """尝试创建融合策略"""
        fusion_result = self.exp_pool.maybe_spawn_fused_strategy(
            fusion_counter=self._fusion_counter,
            use_best=True
        )
        
        if fusion_result:
            new_strategy_id, fusion_hint, UB_new, LB_new, DT_new = fusion_result
            
            # 注册融合策略
            self.registry.register_fused_strategy(new_strategy_id, fusion_hint)
            
            # 记录到搜索状态
            self.state.fused_strategies[new_strategy_id] = fusion_hint
            
            self._fusion_counter += 1
            
            logger.info(
                f"🔬 创建融合策略: {new_strategy_id} "
                f"(UB={UB_new:.2f}, LB={LB_new:.2f}, DT={DT_new:.2f})"
            )


    async def search_best_kernel(self,
                                problem_info: Dict[str, Any],
                                pytorch_code: str,
                                pytorch_forward,
                                test_inputs,
                                init_inputs) -> Dict[str, Any]:
        """
        搜索最佳kernel的主入口
        
        Args:
            problem_info: 问题信息
            pytorch_code: PyTorch代码
            pytorch_forward: PyTorch forward函数
            test_inputs: 测试输入
            init_inputs: 初始化输入
            
        Returns:
            搜索结果摘要
        """
        logger.info("=" * 80)
        logger.info("开始多Worker进化式搜索")
        logger.info(f"配置: {self.config.num_workers} workers, {self.config.max_rounds} rounds")
        logger.info("=" * 80)
        
        # 创建算子和硬件标识
        op_key = self._create_operator_key(problem_info)
        hw = self._create_hardware_signature()
        
        logger.info(f"算子: {op_key}")
        logger.info(f"硬件: {hw}")
        
        # 运行一次analysis_chain（所有worker共享）
        logger.info("\n🔍 运行分析阶段...")
        analysis_result = await self.analysis_chain.analyze_operation(
            pytorch_code=pytorch_code,
            problem_info=problem_info,
            iteration=1
        )
        
        if not analysis_result.get("success", False):
            error_msg = analysis_result.get("error", "分析失败")
            logger.error(f"分析阶段失败: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "op_key": str(op_key),
                "hw": str(hw)
            }
        
        analysis_output = analysis_result["result"]
        logger.info("✅ 分析阶段完成")
        
        # 执行多轮搜索
        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"🔄 轮次 {round_num}/{self.config.max_rounds}")
            logger.info(f"{'=' * 80}")
            
            self.state.round = round_num
            
            # 选择策略
            selected_strategies = self._select_strategies_for_round(
                round_num, op_key, hw
            )
            self.state.active_strategies = selected_strategies
            
            # 执行worker
            round_results = await self._execute_round(
                round_num=round_num,
                strategies=selected_strategies,
                op_key=op_key,
                hw=hw,
                analysis_output=analysis_output,
                pytorch_code=pytorch_code,
                problem_info=problem_info,
                pytorch_forward=pytorch_forward,
                test_inputs=test_inputs,
                init_inputs=init_inputs
            )
            
            # 输出轮次摘要
            self._print_round_summary(round_num, round_results)
        
        # 生成最终结果
        final_result = self._generate_final_result(op_key, hw, problem_info)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 搜索完成!")
        logger.info("=" * 80)
        
        return final_result
    
    async def _execute_round(self,
                           round_num: int,
                           strategies: List[StrategyId],
                           op_key: OperatorKey,
                           hw: HardwareSignature,
                           analysis_output: Dict[str, Any],
                           pytorch_code: str,
                           problem_info: Dict[str, Any],
                           pytorch_forward,
                           test_inputs,
                           init_inputs) -> List[WorkerResult]:
        """
        执行一轮搜索（带资源管理和超时控制）
        
        Args:
            round_num: 轮次编号
            strategies: 策略列表
            op_key: 算子标识符
            hw: 硬件签名
            analysis_output: 分析输出
            pytorch_code: PyTorch代码
            problem_info: 问题信息
            pytorch_forward: PyTorch forward函数
            test_inputs: 测试输入
            init_inputs: 初始化输入
            
        Returns:
            Worker结果列表
        """
        logger.info(f"执行 {len(strategies)} 个worker...")
        
        # 创建worker任务（带超时控制）
        tasks = []
        for i, strategy_id in enumerate(strategies):
            worker_id = f"worker_{round_num}_{i}"
            
            # 包装任务以添加超时和重试
            task = self._run_worker_with_timeout(
                worker_id=worker_id,
                strategy_id=strategy_id,
                op_key=op_key,
                hw=hw,
                analysis_output=analysis_output,
                pytorch_code=pytorch_code,
                problem_info=problem_info,
                pytorch_forward=pytorch_forward,
                test_inputs=test_inputs,
                init_inputs=init_inputs,
                timeout=180  # 3分钟超时
            )
            tasks.append(task)
        
        # 并行执行所有worker
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Worker执行异常: {result}")
                continue
            
            # 处理worker结果
            self._process_worker_result(result, op_key, hw)
            valid_results.append(result)
        
        return valid_results
    
    async def _run_worker_with_timeout(self,
                                      timeout: float = 180,
                                      max_retries: int = 1,
                                      **kwargs) -> WorkerResult:
        """
        运行worker并添加超时和重试机制
        
        Args:
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            **kwargs: run_worker_once的参数
            
        Returns:
            WorkerResult
        """
        worker_id = kwargs.get("worker_id", "unknown")
        strategy_id = kwargs.get("strategy_id", "unknown")
        op_key = kwargs.get("op_key")
        
        for attempt in range(max_retries + 1):
            try:
                # 添加超时控制
                result = await asyncio.wait_for(
                    self.run_worker_once(**kwargs),
                    timeout=timeout
                )
                return result
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"Worker {worker_id} 超时 (尝试 {attempt + 1}/{max_retries + 1})"
                )
                if attempt >= max_retries:
                    return self._create_failed_result(
                        worker_id, strategy_id, op_key,
                        f"执行超时 ({timeout}秒)"
                    )
            
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} 执行失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt >= max_retries:
                    return self._create_failed_result(
                        worker_id, strategy_id, op_key, str(e)
                    )
            
            # 短暂延迟后重试
            await asyncio.sleep(1)
        
        # 不应该到达这里
        return self._create_failed_result(
            worker_id, strategy_id, op_key, "未知错误"
        )
    
    def _print_round_summary(self, round_num: int, results: List[WorkerResult]) -> None:
        """
        打印轮次摘要
        
        Args:
            round_num: 轮次编号
            results: Worker结果列表
        """
        logger.info(f"\n📊 轮次 {round_num} 摘要:")
        logger.info(f"  总评估: {len(results)}")
        
        valid_results = [r for r in results if r.valid]
        logger.info(f"  成功: {len(valid_results)}")
        
        if valid_results:
            best_result = max(valid_results, key=lambda r: r.sigma)
            logger.info(f"  本轮最佳: {best_result.strategy_id} - {best_result.sigma:.3f}x")
        
        # 全局最佳
        global_best = self.state.get_global_best()
        if global_best:
            logger.info(f"  全局最佳: {global_best.strategy_id} - {global_best.sigma:.3f}x")
        
        # 成功率
        success_rate = self.state.calculate_success_rate()
        logger.info(f"  累计成功率: {success_rate:.1%}")
    
    def _generate_final_result(self,
                              op_key: OperatorKey,
                              hw: HardwareSignature,
                              problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成最终搜索结果
        
        Args:
            op_key: 算子标识符
            hw: 硬件签名
            problem_info: 问题信息
            
        Returns:
            最终结果字典
        """
        global_best = self.state.get_global_best()
        
        result = {
            "success": global_best is not None,
            "op_key": str(op_key),
            "hw": str(hw),
            "operation_name": problem_info.get("operation_name", "unknown"),
            "total_rounds": self.state.round,
            "total_evaluations": self.state.total_evaluations,
            "successful_evaluations": self.state.successful_evaluations,
            "success_rate": self.state.calculate_success_rate(),
            "num_strategies_tried": len(self.state.best_sigma),
            "num_fused_strategies": len(self.state.fused_strategies),
            "knowledge_base_size": self.kb.size(),
            "experience_pool_size": self.exp_pool.size()
        }
        
        if global_best:
            result.update({
                "best_strategy": global_best.strategy_id,
                "best_speedup": global_best.sigma,
                "best_kernel": global_best.kernel_code,
                "best_T_torch": global_best.T_torch,
                "best_T_kernel": global_best.T_kernel
            })
            
            logger.info(f"\n🏆 最佳结果:")
            logger.info(f"  策略: {global_best.strategy_id}")
            logger.info(f"  加速比: {global_best.sigma:.3f}x")
            logger.info(f"  PyTorch时间: {global_best.T_torch*1000:.3f}ms")
            logger.info(f"  Triton时间: {global_best.T_kernel*1000:.3f}ms")
        else:
            result["error"] = "未找到有效的kernel实现"
            logger.warning("⚠️  未找到有效的kernel实现")
        
        # 添加策略性能排名
        top_strategies = self.state.get_top_strategies(10)
        result["top_strategies"] = [
            {
                "strategy_id": sid,
                "speedup": self.state.best_sigma[sid]
            }
            for sid in top_strategies
        ]
        
        return result
