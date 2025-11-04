"""
Designer Agent - 负责算法设计和伪代码生成，参考aikg架构
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple

from .base import AgentBase
from ..utils.workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)


class DesignerAgent(AgentBase):
    """
    算法设计Agent
    
    负责分析PyTorch算子需求，设计高效的算法实现方案，
    并生成详细的伪代码和优化建议。
    """
    
    def __init__(self, op_name: str, task_desc: str, dsl: str, backend: str, arch: str,
                 workflow_config_path: str = None, config: dict = None):
        """
        初始化Designer Agent
        
        Args:
            op_name: 算子名称
            task_desc: PyTorch算子代码描述
            dsl: DSL类型 (triton)
            backend: 后端类型 (triton)
            arch: 架构类型 (nvidia)
            workflow_config_path: 工作流配置路径
            config: 配置字典
        """
        # 设置上下文信息
        context = {
            "agent_name": "designer",
            "op_name": op_name,
            "task_desc": task_desc,
            "dsl": dsl,
            "backend": backend,
            "arch": arch
        }
        
        super().__init__(context=context, config=config)
        
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.backend = backend
        self.arch = arch
        
        # 加载工作流配置
        if workflow_config_path:
            self.workflow_config = WorkflowManager.load_workflow_config(workflow_config_path)
        else:
            self.workflow_config = WorkflowManager._get_default_workflow_config()
        
        # 设置基础文档信息
        self.base_doc = {
            "dsl": dsl,
            "backend": backend,
            "arch": arch,
            "agent_type": "designer"
        }
        
        logger.info(f"Designer Agent初始化完成: {op_name}")
    
    async def run(self, task_info: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        运行Designer Agent
        
        Args:
            task_info: 任务信息字典
            
        Returns:
            Tuple[str, str, str]: (结果JSON字符串, 提示词, 推理内容)
        """
        logger.info(f"Designer开始分析算子: {self.op_name}")
        
        try:
            # 加载提示词模板
            # 创建简化的提示词模板，避免文件路径问题
            from langchain_core.prompts import PromptTemplate
            
            template = """你是一个专业的GPU算子算法设计专家，擅长为NVIDIA GPU设计高性能的Triton kernel算法。

## 任务信息
算子名称：{op_name}
DSL类型：{dsl}
后端类型：{backend}
架构类型：{arch}

## PyTorch算子代码
```python
{pytorch_code}
```

## 任务要求

请你作为算法设计专家，分析上述PyTorch算子代码，完成以下任务：

1. **算法分析**：
   - 分析算子的数学原理和计算特点
   - 识别计算瓶颈和优化机会
   - 考虑GPU内存访问模式和并行化策略

2. **设计方案**：
   - 设计适合GPU并行计算的算法流程
   - 确定合适的线程块大小和网格配置
   - 规划内存访问模式和数据布局

3. **伪代码实现**：
   - 提供详细的算法伪代码
   - 标注关键的优化点和注意事项
   - 说明并行化策略和内存管理

4. **优化建议**：
   - 提出针对Triton的具体优化建议
   - 考虑向量化、内存合并、寄存器使用等因素
   - 预估性能特征和可能的瓶颈

请以JSON格式返回你的设计结果：
```json
{{
    "algorithm_analysis": "详细的算法分析，包括数学原理、计算特点、瓶颈识别",
    "design_strategy": "设计策略说明，包括并行化方案、内存布局、线程配置", 
    "pseudocode": "详细的伪代码实现，包含关键优化点和注意事项",
    "optimization_suggestions": [
        "针对Triton的具体优化建议1",
        "针对Triton的具体优化建议2",
        "针对Triton的具体优化建议3"
    ],
    "performance_considerations": "性能考虑因素，包括预期瓶颈和性能特征",
    "memory_pattern": "内存访问模式说明，包括全局内存、共享内存、寄存器使用"
}}
```

## 设计原则
- 重点考虑GPU的并行计算特性和SIMT架构
- 优化内存访问效率，减少全局内存访问延迟
- 充分利用共享内存和寄存器，提高数据局部性
- 考虑不同输入规模的适应性和可扩展性
- 遵循Triton编程最佳实践和NVIDIA GPU架构特点"""

            prompt_template = PromptTemplate(
                template=template,
                input_variables=["op_name", "dsl", "backend", "arch", "pytorch_code"]
            )
            
            # 使用简化的文档内容
            triton_docs = "Triton是一种用于编写高效GPU kernel的Python DSL，支持类似NumPy的语法。"
            optimization_guide = "优化建议：使用向量化操作，优化内存访问模式，合理选择block size。"
            
            # 准备输入参数
            input_params = {
                "op_name": self.op_name,
                "dsl": self.dsl,
                "backend": self.backend,
                "arch": self.arch,
                "pytorch_code": task_info.get("task_desc", self.task_desc)
            }
            
            # 获取模型配置
            model_name = self._get_model_name()
            
            # 调用LLM
            content, formatted_prompt, reasoning_content = await self.run_llm(
                prompt_template, input_params, model_name
            )
            
            # 解析响应
            design_result = self._parse_design_response(content)
            
            # 格式化为JSON字符串
            result_json = json.dumps({"code": design_result}, ensure_ascii=False, indent=2)
            
            logger.info(f"Designer完成算子设计: {self.op_name}")
            return result_json, formatted_prompt, reasoning_content
            
        except Exception as e:
            logger.error(f"Designer执行失败: {str(e)}")
            error_result = {
                "code": f"ERROR: Designer执行失败: {str(e)}"
            }
            return json.dumps(error_result), "", ""
    
    def _get_model_name(self) -> str:
        """获取模型名称"""
        if self.config and "agent_model_config" in self.config:
            agent_config = self.config["agent_model_config"]
            return agent_config.get("designer", agent_config.get("default", "deepseek_default"))
        return "deepseek_default"
    
    def _parse_design_response(self, response: str) -> str:
        """
        解析设计响应
        
        Args:
            response: LLM响应
            
        Returns:
            解析后的设计结果
        """
        try:
            # 尝试提取JSON格式的设计结果
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                design_data = json.loads(json_str)
                
                # 格式化设计结果
                formatted_design = self._format_design_result(design_data)
                return formatted_design
            else:
                # 如果没有JSON格式，直接返回响应内容
                return response.strip()
                
        except json.JSONDecodeError:
            logger.warning("无法解析JSON格式的设计结果，返回原始响应")
            return response.strip()
        except Exception as e:
            logger.error(f"解析设计响应失败: {str(e)}")
            return f"设计解析失败: {str(e)}"
    
    def _format_design_result(self, design_data: Dict[str, Any]) -> str:
        """
        格式化设计结果
        
        Args:
            design_data: 设计数据字典
            
        Returns:
            格式化后的设计结果字符串
        """
        formatted_parts = []
        
        # 算法分析
        if "algorithm_analysis" in design_data:
            formatted_parts.append(f"## 算法分析\n{design_data['algorithm_analysis']}")
        
        # 设计策略
        if "design_strategy" in design_data:
            formatted_parts.append(f"## 设计策略\n{design_data['design_strategy']}")
        
        # 伪代码实现
        if "pseudocode" in design_data:
            formatted_parts.append(f"## 伪代码实现\n```\n{design_data['pseudocode']}\n```")
        
        # 优化建议
        if "optimization_suggestions" in design_data:
            suggestions = design_data["optimization_suggestions"]
            if isinstance(suggestions, list):
                suggestions_text = "\n".join([f"- {s}" for s in suggestions])
            else:
                suggestions_text = str(suggestions)
            formatted_parts.append(f"## 优化建议\n{suggestions_text}")
        
        # 性能考虑
        if "performance_considerations" in design_data:
            formatted_parts.append(f"## 性能考虑\n{design_data['performance_considerations']}")
        
        # 内存访问模式
        if "memory_pattern" in design_data:
            formatted_parts.append(f"## 内存访问模式\n{design_data['memory_pattern']}")
        
        return "\n\n".join(formatted_parts)