"""
Kernel生成器 - 基于langchain的多Agent协作系统
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from ..agents.designer import DesignerAgent
from ..agents.coder import CoderAgent
from ..agents.conductor import ConductorAgent
from .verifier import TritonVerifier
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class KernelGenerator:
    """
    Triton Kernel生成器
    
    协调Designer、Coder、Conductor三个Agent的工作，
    完成从PyTorch算子描述到Triton kernel的完整转换流程。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化生成器
        
        Args:
            config: 配置参数
        """
        self.config_manager = ConfigManager(config)
        self.config = self.config_manager.get_config()
        
        # 初始化Agent
        self.designer = DesignerAgent(self.config)
        self.coder = CoderAgent(self.config)
        self.conductor = ConductorAgent(self.config)
        
        # 初始化验证器
        self.verifier = TritonVerifier(self.config)
        
        logger.info("KernelGenerator初始化完成")
    
    async def generate_kernel(
        self,
        op_name: str,
        task_desc: str,
        input_shapes: List[tuple],
        dtype: str = "float32",
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        生成Triton kernel
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            input_shapes: 输入张量形状
            dtype: 数据类型
            max_iterations: 最大迭代次数
            
        Returns:
            Dict包含生成的kernel代码和性能指标
        """
        logger.info(f"开始生成kernel: {op_name}")
        
        # 构建任务上下文
        context = {
            "op_name": op_name,
            "task_desc": task_desc,
            "input_shapes": input_shapes,
            "dtype": dtype,
            "iteration": 0,
            "max_iterations": max_iterations
        }
        
        # 执行多Agent工作流
        for iteration in range(max_iterations):
            context["iteration"] = iteration
            logger.info(f"开始第 {iteration + 1} 轮迭代")
            
            try:
                # Step 1: Designer设计算法
                design_result = await self._run_designer(context)
                if not design_result["success"]:
                    logger.error(f"Designer执行失败: {design_result.get('error')}")
                    continue
                
                context["design"] = design_result["design"]
                
                # Step 2: Coder生成代码
                code_result = await self._run_coder(context)
                if not code_result["success"]:
                    logger.error(f"Coder执行失败: {code_result.get('error')}")
                    continue
                
                context["kernel_code"] = code_result["kernel_code"]
                
                # Step 3: 验证和性能测试
                verify_result = await self._run_verifier(context)
                if verify_result["success"]:
                    logger.info(f"Kernel生成成功，迭代次数: {iteration + 1}")
                    return {
                        "kernel_code": context["kernel_code"],
                        "design": context["design"],
                        "performance_metrics": verify_result["performance_metrics"],
                        "iterations": iteration + 1
                    }
                
                # Step 4: Conductor决策下一步
                conductor_result = await self._run_conductor(context, verify_result)
                if conductor_result["action"] == "terminate":
                    logger.warning("Conductor决定终止生成流程")
                    break
                
                # 根据Conductor建议更新上下文
                context.update(conductor_result.get("suggestions", {}))
                
            except Exception as e:
                logger.error(f"第 {iteration + 1} 轮迭代出现异常: {str(e)}")
                continue
        
        # 如果所有迭代都失败
        logger.error(f"Kernel生成失败，已达到最大迭代次数 {max_iterations}")
        return {
            "kernel_code": context.get("kernel_code"),
            "design": context.get("design"),
            "performance_metrics": None,
            "iterations": max_iterations,
            "success": False,
            "error": "达到最大迭代次数仍未生成有效kernel"
        }
    
    async def _run_designer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行Designer Agent"""
        start_time = time.time()
        try:
            result = await self.designer.design_algorithm(
                op_name=context["op_name"],
                task_desc=context["task_desc"],
                input_shapes=context["input_shapes"],
                dtype=context["dtype"]
            )
            execution_time = time.time() - start_time
            logger.info(f"Designer执行完成，耗时 {execution_time:.2f}秒")
            return {"success": True, "design": result, "execution_time": execution_time}
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Designer执行失败: {str(e)}")
            return {"success": False, "error": str(e), "execution_time": execution_time}
    
    async def _run_coder(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行Coder Agent"""
        start_time = time.time()
        try:
            result = await self.coder.generate_code(
                design=context["design"],
                op_name=context["op_name"],
                input_shapes=context["input_shapes"],
                dtype=context["dtype"]
            )
            execution_time = time.time() - start_time
            logger.info(f"Coder执行完成，耗时 {execution_time:.2f}秒")
            return {"success": True, "kernel_code": result, "execution_time": execution_time}
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Coder执行失败: {str(e)}")
            return {"success": False, "error": str(e), "execution_time": execution_time}
    
    async def _run_verifier(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行验证器"""
        start_time = time.time()
        try:
            result = await self.verifier.verify_kernel(
                kernel_code=context["kernel_code"],
                op_name=context["op_name"],
                input_shapes=context["input_shapes"],
                dtype=context["dtype"]
            )
            execution_time = time.time() - start_time
            logger.info(f"验证器执行完成，耗时 {execution_time:.2f}秒")
            return {
                "success": result["success"],
                "performance_metrics": result.get("performance_metrics"),
                "error": result.get("error"),
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"验证器执行失败: {str(e)}")
            return {"success": False, "error": str(e), "execution_time": execution_time}
    
    async def _run_conductor(self, context: Dict[str, Any], verify_result: Dict[str, Any]) -> Dict[str, Any]:
        """运行Conductor Agent"""
        start_time = time.time()
        try:
            result = await self.conductor.make_decision(
                context=context,
                verify_result=verify_result
            )
            execution_time = time.time() - start_time
            logger.info(f"Conductor执行完成，耗时 {execution_time:.2f}秒")
            return {"action": result["action"], "suggestions": result.get("suggestions", {})}
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Conductor执行失败: {str(e)}")
            return {"action": "terminate", "error": str(e)}