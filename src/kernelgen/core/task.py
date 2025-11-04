"""
任务管理模块 - 基于langchain的多Agent任务编排，参考aikg架构
"""

import os
import json
import logging
import asyncio
import time
from typing import Tuple, Optional, Dict, Any, List
from .async_pool.device_pool import DevicePool
from ..utils.workflow_manager import WorkflowManager
from ..agents.conductor import ConductorAgent
from ..agents.coder import CoderAgent
from ..agents.designer import DesignerAgent
from .verifier import TritonVerifier
from ..utils.collector import get_collector

def get_project_root():
    """获取项目根目录的绝对路径"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


class Task:
    """
    配置驱动的任务类，基于workflow.yaml进行工作流管理
    参考aikg的Task实现，适配PyTorch + Triton + NVIDIA GPU
    """

    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 task_id: str,
                 config: dict,
                 device_pool: DevicePool,
                 dtype: str = "float32",
                 workflow: Optional[str] = None,
                 inspirations: Optional[List[str]] = None,
                 meta_prompts: Optional[str] = None) -> None:
        """
        初始化Task类，基于workflow配置进行工作流管理。

        Args:
            op_name (str): 算子名称。
            task_desc (str): PyTorch算子代码描述。
            task_id (str): 任务ID。
            config (dict): 配置agent_model_config。
            device_pool: 设备池。
            dtype (str, optional): 数据类型, 默认为"float32"。
            workflow (str, optional): workflow名称，可以是文件名或完整路径。
            inspirations (List[str], optional): 启发示例列表。
        """
        # 基础属性
        self.op_name = op_name
        self.task_desc = task_desc  # PyTorch算子代码
        self.task_id = task_id
        self.backend = "triton"  # 固定为triton
        self.arch = "nvidia"     # 固定为nvidia
        self.dsl = "triton"      # 固定为triton
        self.framework = "pytorch"  # 固定为pytorch
        self.dtype = dtype
        self.device_pool = device_pool
        self.inspirations = inspirations
        self.meta_prompts = meta_prompts

        # 统一保存config，后续向下传递
        self.config = config

        # 优先使用传入的workflow参数，否则使用config中的workflow_config_path
        if workflow:
            self.workflow_config_path = WorkflowManager.resolve_workflow_config_path(workflow)
        else:
            self.workflow_config_path = config.get("workflow_config_path")
            if self.workflow_config_path and not os.path.isabs(self.workflow_config_path):
                # 相对路径需要相对于项目根目录
                self.workflow_config_path = os.path.join(get_project_root(), self.workflow_config_path)

        # 确保workflow_config_path不为空
        if not self.workflow_config_path:
            raise ValueError("workflow_config_path is required. Please provide it in config or as workflow parameter.")

        # 初始化Conductor（优先初始化，用于获取workflow配置）
        self.conductor = ConductorAgent(
            op_name=self.op_name,
            task_desc=self.task_desc,
            task_id=self.task_id,
            dsl=self.dsl,
            framework=self.framework,
            arch=self.arch,
            workflow_config_path=self.workflow_config_path,
            config=self.config
        )

        # 根据workflow配置动态初始化agents
        self._init_agents_from_workflow()

    def _init_agents_from_workflow(self):
        """根据workflow.yaml配置动态初始化需要的agents"""
        # 获取workflow中配置的agent列表
        agent_names = set(self.conductor.agent_info.keys())

        # 初始化所需的agents
        self.agents = {}  # 存储所有agents的字典

        # 根据配置创建相应的agents
        if 'designer' in agent_names:
            self.designer = DesignerAgent(self.op_name, self.task_desc,
                                         self.dsl, self.backend, self.arch,
                                         workflow_config_path=self.workflow_config_path, config=self.config)
            self.agents['designer'] = self.designer

        if 'coder' in agent_names:
            self.coder = CoderAgent(self.op_name, self.task_desc,
                                   self.dsl, self.framework, self.backend, self.arch,
                                   workflow_config_path=self.workflow_config_path, config=self.config)
            self.agents['coder'] = self.coder

        if 'verifier' in agent_names:
            self.verifier = TritonVerifier(self.op_name, self.task_desc,
                                          self.task_id, self.framework, self.dsl, self.backend, self.arch, config=self.config)
            self.agents['verifier'] = self.verifier

    def get_agent(self, agent_name: str):
        """获取指定名称的agent实例"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' is not available in current workflow configuration. "
                             f"Available agents: {list(self.agents.keys())}")
        return self.agents[agent_name]

    def init_conductor(self, init_task_info: Optional[Dict[str, Any]] = None):
        """
        初始化Conductor，根据初始任务信息进行初始化。

        Args:
            init_task_info (Dict[str, Any], optional): 初始任务信息字典, 包含初始代码等
        """

        # 初始化基础文档
        base_doc = {"backend": self.backend, "arch": self.arch, "dsl": self.dsl, "framework": self.framework}

        # 添加workflow名称
        workflow_name = os.path.basename(self.workflow_config_path).replace(
            '.yaml', '') if self.workflow_config_path else ""
        base_doc["workflow_name"] = workflow_name

        # 只在相应agent存在时添加其基础文档
        if 'designer' in self.agents:
            base_doc.update(self.agents['designer'].base_doc)

        if 'coder' in self.agents:
            coder = self.agents['coder']
            base_doc.update(coder.base_doc)

        # 初始化任务信息
        self.conductor.set_task_info(base_doc)

        # inspirations and meta_prompts from evolution
        self.conductor.task_info.update({"inspirations": self.inspirations})
        self.conductor.task_info.update({"meta_prompts": self.meta_prompts})

        # 插入初始记录（如果有初始代码）
        if init_task_info and init_task_info.get("designer_code"):
            self.conductor.record_agent_execution(
                agent_name="designer",
                result=json.dumps({"code": init_task_info.get("designer_code")})
            )

        if init_task_info and init_task_info.get("coder_code"):
            self.conductor.record_agent_execution(
                agent_name="coder",
                result=json.dumps({"code": init_task_info.get("coder_code")})
            )

    async def run(self, init_task_info: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, dict]:
        """
        异步运行任务，执行PyTorch算子到Triton kernel的转换

        Args:
            init_task_info: 初始任务信息字典，包含初始代码等

        Returns:
            Tuple[str, bool, dict]: (算子名称, 是否成功, 任务信息)
        """
        try:
            # 初始化conductor
            self.init_conductor(init_task_info)

            # 获取首个agent（通过yaml配置）
            current_agent = self.conductor.start_agent

            while current_agent != "finish":
                logger.info(f"Task {self.task_id}, op_name: {self.op_name}, current_agent: {current_agent}")
                try:
                    if current_agent == "designer":
                        designer = self.get_agent('designer')

                        designer_res, designer_prompt, designer_reasoning = await designer.run(
                            task_info=self.conductor.task_info
                        )
                        self.conductor.record_agent_execution(
                            agent_name="designer",
                            result=designer_res,
                            prompt=designer_prompt,
                            reasoning=designer_reasoning
                        )

                    elif current_agent == "coder":
                        coder = self.get_agent('coder')

                        coder_res, coder_prompt, coder_reasoning = await coder.run(
                            task_info=self.conductor.task_info
                        )

                        self.conductor.record_agent_execution(
                            agent_name="coder",
                            result=coder_res,
                            prompt=coder_prompt,
                            reasoning=coder_reasoning
                        )

                    elif current_agent == "verifier":
                        device_id = await self.device_pool.acquire_device()
                        try:
                            current_step = len(self.conductor.trace.trace_list)
                            loop = asyncio.get_running_loop()
                            # 传递task_info而不是parsed_code
                            verify_res, verify_log = await loop.run_in_executor(
                                None,
                                self.verifier.run,
                                self.conductor.task_info, current_step, device_id
                            )
                            profile_res = ()
                            # 针对NVIDIA GPU的性能分析
                            if verify_res and self.backend == "triton":
                                profile_settings = self.config.get("profile_settings", {})
                                profile_res = await loop.run_in_executor(
                                    None,
                                    self.verifier.run_profile,
                                    current_step, device_id, profile_settings
                                )

                            self.conductor.record_agent_execution(
                                agent_name="verifier",
                                result=str(verify_res),
                                error_log=verify_log,
                                profile_res=profile_res
                            )
                        finally:
                            await self.device_pool.release_device(device_id)
                    else:
                        raise ValueError(f"Unsupported agent: {current_agent}")

                    # 获取下一个agent
                    next_agent = await self.conductor.get_next_agent()
                    current_agent = next_agent

                except Exception as agent_error:
                    logger.error(f"Agent {current_agent} execution failed: {agent_error}")
                    logger.error(f"Error type: {type(agent_error).__name__}")
                    logger.error(f"Full error details: {repr(agent_error)}")

                    self.conductor.record_agent_execution(
                        agent_name=current_agent,
                        result=f"ERROR: Agent execution failed: {str(agent_error)}",
                        error_log=str(agent_error)
                    )
                    return self.op_name, False, self.conductor.task_info

            # 获取最终结果
            final_success = self.conductor.task_info.get('verifier_result', False)

            # 数据收集（如果启用）
            if os.getenv("KERNELGEN_DATA_COLLECT", "off").lower() == "on":
                try:
                    collector = await get_collector()
                    collector.set_config(self.config)

                    # 根据验证结果准备数据
                    if final_success:
                        # 验证成功：收集task相关数据 + database数据
                        saved_files = await collector.prepare_and_remove_data(task_id=self.task_id)
                        database_file = collector.prepare_database_data(self.conductor.task_info)
                        all_files = saved_files + ([database_file] if database_file else [])
                        logger.debug(
                            f"Task {self.task_id} completed successfully, saved {len(all_files)} files: {all_files}")
                    else:
                        # 验证失败：收集无task_id的独立数据
                        saved_files = await collector.prepare_and_remove_data()
                        logger.debug(f"Task {self.task_id} failed, saved {len(saved_files)} files: {saved_files}")
                except Exception as e:
                    logger.error(f"Failed to prepare data for transmission in task {self.task_id}: {e}")

            return self.op_name, final_success, self.conductor.task_info

        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return self.op_name, False, self.conductor.task_info