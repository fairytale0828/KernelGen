"""
Conductor Agent - 负责任务编排和决策，参考aikg架构
"""

import logging
import json
from typing import Dict, Any, Optional

from ..utils.workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)


class ConductorAgent:
    """
    任务编排Agent
    
    负责分析验证结果，决定下一步行动，
    并提供优化建议来改进kernel生成质量。
    """
    
    def __init__(self, op_name: str, task_desc: str, task_id: str, dsl: str, 
                 framework: str, arch: str, workflow_config_path: str = None, 
                 config: dict = None):
        """
        初始化Conductor Agent
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            task_id: 任务ID
            dsl: DSL类型
            framework: 框架类型
            arch: 架构类型
            workflow_config_path: 工作流配置路径
            config: 配置字典
        """
        self.op_name = op_name
        self.task_desc = task_desc
        self.task_id = task_id
        self.dsl = dsl
        self.framework = framework
        self.arch = arch
        self.config = config or {}
        
        # 加载工作流配置
        if workflow_config_path:
            self.workflow_config = WorkflowManager.load_workflow_config(workflow_config_path)
        else:
            self.workflow_config = WorkflowManager._get_default_workflow_config()
        
        # 验证工作流配置
        WorkflowManager.validate_workflow_config(self.workflow_config)
        
        # 提取工作流信息
        self.agent_info = self.workflow_config.get("agent_info", {})
        self.start_agent = self.workflow_config.get("start_agent", "designer")
        self.mandatory_llm_analysis = self.workflow_config.get("mandatory_llm_analysis", [])
        self.limitation_info = self.workflow_config.get("limitation_info", {})
        
        # 初始化任务信息和追踪
        self.task_info = {}
        self.trace = SimpleTrace()
        
        logger.info(f"Conductor Agent初始化完成: {op_name}")
    
    def set_task_info(self, task_info: Dict[str, Any]):
        """设置任务信息"""
        self.task_info.update(task_info)
    
    def record_agent_execution(self, agent_name: str, result: str, 
                             prompt: str = "", reasoning: str = "", 
                             error_log: str = "", profile_res: tuple = ()):
        """记录Agent执行结果"""
        execution_record = {
            "agent_name": agent_name,
            "result": result,
            "prompt": prompt,
            "reasoning": reasoning,
            "error_log": error_log,
            "profile_res": profile_res
        }
        
        self.trace.add_record(execution_record)
        
        # 更新任务信息
        if agent_name == "designer":
            self.task_info["designer_result"] = result
        elif agent_name == "coder":
            self.task_info["coder_result"] = result
        elif agent_name == "verifier":
            self.task_info["verifier_result"] = result == "True"
            if profile_res:
                self.task_info["performance_metrics"] = profile_res
        
        logger.debug(f"记录Agent执行: {agent_name}")
    
    async def get_next_agent(self) -> str:
        """获取下一个要执行的Agent"""
        current_step = len(self.trace.trace_list)
        max_steps = self.limitation_info.get("required", {}).get("max_step", 20)
        
        # 检查是否达到最大步数
        if current_step >= max_steps:
            logger.warning(f"达到最大步数限制: {max_steps}")
            return "finish"
        
        # 获取最后执行的Agent
        if not self.trace.trace_list:
            return self.start_agent
        
        last_record = self.trace.trace_list[-1]
        last_agent = last_record["agent_name"]
        
        # 根据工作流配置决定下一个Agent
        if last_agent in self.agent_info:
            possible_next = self.agent_info[last_agent].get("possible_next_agent", ["finish"])
            
            # 简化决策逻辑
            if len(possible_next) == 1:
                return possible_next[0]
            
            # 如果有多个选择，根据验证结果决定
            if last_agent == "verifier":
                verifier_success = self.task_info.get("verifier_result", False)
                if verifier_success:
                    return "finish"
                else:
                    # 验证失败，重新生成代码
                    return "coder" if "coder" in possible_next else "finish"
            
            # 默认选择第一个
            return possible_next[0]
        
        return "finish"


class SimpleTrace:
    """简化的追踪类"""
    
    def __init__(self):
        self.trace_list = []
    
    def add_record(self, record: Dict[str, Any]):
        """添加执行记录"""
        self.trace_list.append(record)