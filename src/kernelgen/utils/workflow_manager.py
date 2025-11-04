"""
工作流管理器 - 管理Agent执行流程
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
def get_project_root():
    """获取项目根目录的绝对路径"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    工作流管理器
    
    负责加载和管理Agent执行工作流配置，
    控制Agent的执行顺序和条件。
    """
    
    @staticmethod
    def resolve_workflow_config_path(workflow: str) -> str:
        """
        解析工作流配置文件路径
        
        Args:
            workflow: 工作流名称或路径
            
        Returns:
            完整的配置文件路径
        """
        if os.path.isabs(workflow):
            return workflow
        
        # 如果是相对路径或文件名，在config目录下查找
        project_root = get_project_root()
        
        # 尝试不同的路径组合
        possible_paths = [
            os.path.join(project_root, "config", f"{workflow}.yaml"),
            os.path.join(project_root, "config", workflow),
            os.path.join(project_root, workflow)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都找不到，返回默认路径
        default_path = os.path.join(project_root, "config", "default_workflow.yaml")
        logger.warning(f"工作流配置文件未找到: {workflow}, 使用默认配置: {default_path}")
        return default_path
    
    @staticmethod
    def load_workflow_config(config_path: str) -> Dict[str, Any]:
        """
        加载工作流配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            工作流配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"加载工作流配置: {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"工作流配置文件不存在: {config_path}")
            return WorkflowManager._get_default_workflow_config()
        except yaml.YAMLError as e:
            logger.error(f"工作流配置文件格式错误: {e}")
            return WorkflowManager._get_default_workflow_config()
        except Exception as e:
            logger.error(f"加载工作流配置失败: {e}")
            return WorkflowManager._get_default_workflow_config()
    
    @staticmethod
    def _get_default_workflow_config() -> Dict[str, Any]:
        """获取默认工作流配置"""
        return {
            "agent_info": {
                "designer": {
                    "possible_next_agent": ["coder"],
                    "output_format": {
                        "parser_name": "designer_parser",
                        "parser_definition": {
                            "output_fields": {
                                "code": {
                                    "field_type": "str",
                                    "mandatory": True,
                                    "field_description": "Algorithm pseudocode implementation"
                                }
                            }
                        }
                    }
                },
                "coder": {
                    "possible_next_agent": ["verifier"],
                    "output_format": {
                        "parser_name": "coder_parser",
                        "parser_definition": {
                            "output_fields": {
                                "code": {
                                    "field_type": "str",
                                    "mandatory": True,
                                    "field_description": "Complete executable Triton kernel code"
                                }
                            }
                        }
                    }
                },
                "verifier": {
                    "possible_next_agent": ["finish", "coder"]
                }
            },
            "start_agent": "designer",
            "mandatory_llm_analysis": ["verifier"],
            "limitation_info": {
                "required": {
                    "max_step": 20
                }
            }
        }
    
    @staticmethod
    def validate_workflow_config(config: Dict[str, Any]) -> bool:
        """
        验证工作流配置的有效性
        
        Args:
            config: 工作流配置
            
        Returns:
            配置是否有效
        """
        try:
            # 检查必要字段
            required_fields = ["agent_info", "start_agent"]
            for field in required_fields:
                if field not in config:
                    logger.error(f"工作流配置缺少必要字段: {field}")
                    return False
            
            # 检查起始Agent是否存在
            start_agent = config["start_agent"]
            if start_agent not in config["agent_info"]:
                logger.error(f"起始Agent不存在: {start_agent}")
                return False
            
            # 检查Agent链的完整性
            agent_info = config["agent_info"]
            for agent_name, agent_config in agent_info.items():
                if "possible_next_agent" not in agent_config:
                    logger.error(f"Agent {agent_name} 缺少 possible_next_agent 配置")
                    return False
                
                # 检查下一个Agent是否存在（除了finish）
                for next_agent in agent_config["possible_next_agent"]:
                    if next_agent != "finish" and next_agent not in agent_info:
                        logger.error(f"Agent {agent_name} 的下一个Agent不存在: {next_agent}")
                        return False
            
            logger.info("工作流配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"工作流配置验证失败: {e}")
            return False