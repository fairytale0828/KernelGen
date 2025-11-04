"""
Coder Agent - 负责Triton kernel代码生成，参考aikg架构
"""

import logging
import re
import json
from typing import Dict, Any, List, Tuple

from .base import AgentBase
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class CoderAgent(AgentBase):
    """
    代码生成Agent
    
    负责根据Designer的算法设计，生成高质量的Triton kernel代码，
    包括完整的函数实现和PyTorch集成接口。
    """
    
    def __init__(self, op_name: str, task_desc: str, dsl: str, framework: str, 
                 backend: str, arch: str, workflow_config_path: str = None, 
                 config: dict = None):
        """
        初始化Coder Agent
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            dsl: DSL类型
            framework: 框架类型
            backend: 后端类型
            arch: 架构类型
            workflow_config_path: 工作流配置路径
            config: 配置字典
        """
        # 设置上下文信息
        context = {
            "agent_name": "coder",
            "op_name": op_name,
            "task_desc": task_desc,
            "dsl": dsl,
            "framework": framework,
            "backend": backend,
            "arch": arch
        }
        
        super().__init__(context=context, config=config)
        
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        
        # 设置基础文档信息
        self.base_doc = {
            "dsl": dsl,
            "framework": framework,
            "backend": backend,
            "arch": arch,
            "agent_type": "coder"
        }
        
        logger.info(f"Coder Agent初始化完成: {op_name}")
    
    async def run(self, task_info: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        运行Coder Agent
        
        Args:
            task_info: 任务信息字典
            
        Returns:
            Tuple[str, str, str]: (结果JSON字符串, 提示词, 推理内容)
        """
        logger.info(f"Coder开始生成代码: {self.op_name}")
        
        try:
            # 创建简化的提示词模板
            template = """你是一个专业的Triton kernel开发专家，擅长将算法设计转换为高性能的Triton代码。

## 任务信息
算子名称：{op_name}
DSL类型：{dsl}
框架类型：{framework}
后端类型：{backend}
架构类型：{arch}

## PyTorch算子代码
```python
{pytorch_code}
```

## 设计方案
{design_result}

## 任务要求

请你根据上述信息生成完整的Triton kernel代码。

要求：
1. **完整的Triton kernel实现**：
   - 使用@triton.jit装饰器
   - 正确的参数定义和类型注解
   - 完整的kernel函数实现
   - 适当的错误处理

2. **PyTorch集成接口**：
   - 提供wrapper函数用于PyTorch调用
   - 正确的张量形状和设备处理
   - 合适的grid配置和block size

3. **性能优化**：
   - 使用向量化操作
   - 优化内存访问模式
   - 合理使用共享内存

请生成完整可执行的代码：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def {op_name}_kernel(
    # 在这里定义kernel参数
):
    # 在这里实现kernel逻辑
    pass

def {op_name}_triton(
    # 在这里定义wrapper函数参数
):
    # 在这里实现PyTorch接口
    pass
```

注意事项：
- 确保代码语法正确，可以直接运行
- 充分利用Triton的并行计算能力
- 注意内存边界检查，避免越界访问
- 考虑不同输入规模的兼容性"""

            prompt_template = PromptTemplate(
                template=template,
                input_variables=["op_name", "dsl", "framework", "backend", "arch", "pytorch_code", "design_result"]
            )
            
            # 准备输入参数
            design_result = task_info.get("designer_result", "无设计方案")
            
            input_params = {
                "op_name": self.op_name,
                "dsl": self.dsl,
                "framework": self.framework,
                "backend": self.backend,
                "arch": self.arch,
                "pytorch_code": task_info.get("task_desc", self.task_desc),
                "design_result": design_result
            }
            
            # 获取模型配置
            model_name = self._get_model_name()
            
            # 调用LLM
            content, formatted_prompt, reasoning_content = await self.run_llm(
                prompt_template, input_params, model_name
            )
            
            # 解析响应，提取代码
            kernel_code = self._extract_code(content)
            
            # 格式化为JSON字符串
            result_json = json.dumps({"code": kernel_code}, ensure_ascii=False, indent=2)
            
            logger.info(f"Coder完成代码生成: {self.op_name}")
            return result_json, formatted_prompt, reasoning_content
            
        except Exception as e:
            logger.error(f"Coder执行失败: {str(e)}")
            error_result = {
                "code": f"ERROR: Coder执行失败: {str(e)}"
            }
            return json.dumps(error_result), "", ""
    
    def _get_model_name(self) -> str:
        """获取模型名称"""
        if self.config and "agent_model_config" in self.config:
            agent_config = self.config["agent_model_config"]
            return agent_config.get("coder", agent_config.get("default", "deepseek_coder"))
        return "deepseek_coder"
    
    def _extract_code(self, response: str) -> str:
        """
        从响应中提取代码
        
        Args:
            response: LLM响应
            
        Returns:
            提取的代码
        """
        try:
            # 提取Python代码块
            code_pattern = r'```python\s*(.*?)\s*```'
            matches = re.findall(code_pattern, response, re.DOTALL)
            
            if matches:
                # 取最长的代码块（通常是完整实现）
                kernel_code = max(matches, key=len)
                logger.info("成功提取Triton kernel代码")
                return kernel_code.strip()
            else:
                # 如果没有找到代码块，尝试提取整个响应中的代码部分
                lines = response.split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if 'import' in line or '@triton.jit' in line or 'def ' in line:
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                
                if code_lines:
                    kernel_code = '\n'.join(code_lines)
                    logger.info("通过启发式方法提取代码")
                    return kernel_code
                else:
                    logger.warning("未能提取到有效代码，返回原始响应")
                    return response
                    
        except Exception as e:
            logger.error(f"代码提取失败: {str(e)}")
            return response
    
    def _load_prompt_template(self) -> PromptTemplate:
        """加载Coder提示词模板"""
        template = """你是一个专业的Triton kernel开发专家，擅长将算法设计转换为高性能的Triton代码。

算法设计信息：
算子名称：{op_name}
输入张量形状：{input_shapes}
数据类型：{dtype}

设计方案：
{design_strategy}

算法伪代码：
{pseudocode}

优化建议：
{optimization_suggestions}

内存访问模式：
{memory_pattern}

请你作为Triton开发专家，根据上述算法设计生成完整的Triton kernel代码。

要求：
1. **完整的Triton kernel实现**：
   - 使用@triton.jit装饰器
   - 正确的参数定义和类型注解
   - 完整的kernel函数实现
   - 适当的错误处理

2. **PyTorch集成接口**：
   - 提供wrapper函数用于PyTorch调用
   - 正确的张量形状和设备处理
   - 合适的grid配置和block size

3. **性能优化**：
   - 实现算法设计中的优化建议
   - 使用向量化操作
   - 优化内存访问模式
   - 合理使用共享内存

4. **代码质量**：
   - 清晰的注释说明
   - 合理的变量命名
   - 符合Triton编程规范

请生成完整可执行的代码，包含以下部分：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def {op_name}_kernel(
    # 在这里定义kernel参数
):
    # 在这里实现kernel逻辑
    pass

def {op_name}_triton(
    # 在这里定义wrapper函数参数
):
    # 在这里实现PyTorch接口
    pass

# 测试代码（可选）
if __name__ == "__main__":
    # 简单的测试示例
    pass
```

注意事项：
- 确保代码语法正确，可以直接运行
- 充分利用Triton的并行计算能力
- 注意内存边界检查，避免越界访问
- 考虑不同输入规模的兼容性
- 添加必要的性能优化技巧"""

        return PromptTemplate(
            template=template,
            input_variables=[
                "op_name", "input_shapes", "dtype", "design_strategy", 
                "pseudocode", "optimization_suggestions", "memory_pattern"
            ]
        )
    
    async def _process_llm_response(self, response: str) -> str:
        """处理LLM响应，提取Triton kernel代码"""
        try:
            # 提取Python代码块
            code_pattern = r'```python\s*(.*?)\s*```'
            matches = re.findall(code_pattern, response, re.DOTALL)
            
            if matches:
                # 取最长的代码块（通常是完整实现）
                kernel_code = max(matches, key=len)
                logger.info("成功提取Triton kernel代码")
                return kernel_code.strip()
            else:
                # 如果没有找到代码块，尝试提取整个响应中的代码部分
                lines = response.split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if 'import' in line or '@triton.jit' in line or 'def ' in line:
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                
                if code_lines:
                    kernel_code = '\n'.join(code_lines)
                    logger.info("通过启发式方法提取代码")
                    return kernel_code
                else:
                    logger.warning("未能提取到有效代码，返回原始响应")
                    return response
                    
        except Exception as e:
            logger.error(f"代码提取失败: {str(e)}")
            return response
    
    async def generate_code(
        self,
        design: Dict[str, Any],
        op_name: str,
        input_shapes: List[tuple],
        dtype: str = "float32"
    ) -> str:
        """
        生成Triton kernel代码
        
        Args:
            design: Designer的算法设计结果
            op_name: 算子名称
            input_shapes: 输入张量形状
            dtype: 数据类型
            
        Returns:
            生成的Triton kernel代码
        """
        logger.info(f"开始生成Triton代码: {op_name}")
        
        # 格式化输入形状
        shapes_str = ", ".join([str(shape) for shape in input_shapes])
        
        # 格式化优化建议
        opt_suggestions = design.get("optimization_suggestions", [])
        if isinstance(opt_suggestions, list):
            opt_str = "\n".join([f"- {suggestion}" for suggestion in opt_suggestions])
        else:
            opt_str = str(opt_suggestions)
        
        # 执行Agent
        result = await self.execute(
            op_name=op_name,
            input_shapes=shapes_str,
            dtype=dtype,
            design_strategy=design.get("design_strategy", ""),
            pseudocode=design.get("pseudocode", ""),
            optimization_suggestions=opt_str,
            memory_pattern=design.get("memory_pattern", "")
        )
        
        if result["success"]:
            logger.info(f"Triton代码生成完成: {op_name}")
            return result["result"]
        else:
            logger.error(f"代码生成失败: {result['error']}")
            raise Exception(f"Coder执行失败: {result['error']}")