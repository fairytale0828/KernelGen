"""
Debug Agent - 调试专家
负责分析代码错误并提供修复建议
"""

import logging
import re
from typing import Dict, Any, Tuple, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DebugAgent(BaseAgent):
    """
    调试Agent - 负责分析错误并提供修复方案
    
    职责：
    1. 分析编译错误、运行时错误、正确性错误
    2. 提供具体的修复建议
    3. 识别错误根因和解决方案
    4. 为Design Agent提供修复指导
    """
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__("DebugAgent", llm_client, config)
        
        # 错误模式识别
        self.error_patterns = {
            "compilation_error": {
                "import_error": [r"ModuleNotFoundError.*triton", r"ImportError.*triton"],
                "syntax_error": [r"SyntaxError", r"IndentationError"],
                "decorator_error": [r"@triton\.jit.*error", r"triton\.jit.*not found"],
                "type_error": [r"tl\.constexpr.*error", r"constexpr.*not defined"]
            },
            "runtime_error": {
                "cuda_error": [r"CUDA.*error", r"RuntimeError.*CUDA"],
                "shape_error": [r"shape.*mismatch", r"size.*mismatch"],
                "memory_error": [r"out of memory", r"memory.*error"],
                "kernel_launch_error": [r"kernel.*launch.*error", r"grid.*error"]
            },
            "correctness_error": {
                "numerical_error": [r"nan.*detected", r"inf.*detected"],
                "logic_error": [r"assertion.*failed", r"incorrect.*result"],
                "precision_error": [r"precision.*loss", r"floating.*point"]
            }
        }
    
    async def run(self, task_info: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        执行错误分析
        
        Args:
            task_info: 任务信息字典，包含错误信息和代码
            
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        try:
            # 从task_info中获取错误信息
            error_info = task_info.get("error_info", {})
            failed_code = task_info.get("failed_code", "")
            pytorch_code = task_info.get("pytorch_code", "")
            
            # 分类错误类型
            error_category = self._categorize_error(error_info)
            
            # 分析错误根因
            root_cause_analysis = self._analyze_root_cause(error_info, failed_code, error_category)
            
            # 生成修复建议
            fix_suggestions = await self._generate_fix_suggestions(
                error_info, failed_code, pytorch_code, root_cause_analysis
            )
            
            # 构造返回结果
            result_content = {
                "error_analysis": {
                    "error_category": error_category,
                    "root_cause": root_cause_analysis,
                    "fix_suggestions": fix_suggestions
                }
            }
            
            # 格式化为JSON字符串
            import json
            formatted_result = json.dumps(result_content, indent=2, ensure_ascii=False)
            
            return formatted_result, "", ""
            
        except Exception as e:
            logger.error(f"错误分析失败: {str(e)}")
            raise

    
    def _categorize_error(self, error_info: Dict[str, Any]) -> str:
        """分类错误类型"""
        
        error_message = error_info.get("error_message", "")
        error_type = error_info.get("error_type", "unknown")
        
        # 基于错误类型的初步分类
        if error_type == "compilation_error":
            return self._categorize_compilation_error(error_message)
        elif error_type == "runtime_error":
            return self._categorize_runtime_error(error_message)
        elif error_type == "correctness_error":
            return "correctness_error"
        else:
            return "unknown_error"
    
    def _categorize_compilation_error(self, error_message: str) -> str:
        """分类编译错误"""
        
        for error_subtype, patterns in self.error_patterns["compilation_error"].items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return error_subtype
        
        return "compilation_error"
    
    def _categorize_runtime_error(self, error_message: str) -> str:
        """分类运行时错误"""
        
        for error_subtype, patterns in self.error_patterns["runtime_error"].items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return error_subtype
        
        return "runtime_error"
    
    def _analyze_root_cause(self, error_info: Dict[str, Any], failed_code: str, 
                          error_category: str) -> Dict[str, Any]:
        """分析错误根因"""
        
        error_message = error_info.get("error_message", "")
        
        # 基于错误类别分析根因
        if error_category == "import_error":
            primary_cause = "missing_imports"
            fix_strategy = "add_required_imports"
        elif error_category == "syntax_error":
            primary_cause = "syntax_issues"
            fix_strategy = "fix_syntax"
        elif error_category == "decorator_error":
            primary_cause = "triton_decorator_issues"
            fix_strategy = "fix_triton_decorators"
        elif error_category == "cuda_error":
            primary_cause = "cuda_compatibility"
            fix_strategy = "fix_cuda_issues"
        elif error_category == "shape_error":
            primary_cause = "tensor_shape_mismatch"
            fix_strategy = "fix_tensor_shapes"
        elif error_category == "correctness_error":
            primary_cause = "algorithm_logic_error"
            fix_strategy = "fix_algorithm_logic"
        else:
            primary_cause = "unknown_issue"
            fix_strategy = "comprehensive_review"
        
        # 代码分析
        code_issues = self._analyze_code_issues(failed_code, error_category)
        
        return {
            "primary_cause": primary_cause,
            "fix_strategy": fix_strategy,
            "error_category": error_category,
            "code_issues": code_issues,
            "error_severity": self._assess_error_severity(error_category, error_message)
        }
    
    def _analyze_code_issues(self, failed_code: str, error_category: str) -> List[str]:
        """分析代码中的具体问题"""
        
        issues = []
        
        # 检查导入语句
        if "import torch" not in failed_code:
            issues.append("缺少torch导入")
        if "import triton" not in failed_code:
            issues.append("缺少triton导入")
        if "import triton.language as tl" not in failed_code:
            issues.append("缺少triton.language导入")
        
        # 检查装饰器
        if "@triton.jit" not in failed_code and "triton" in failed_code:
            issues.append("缺少@triton.jit装饰器")
        
        # 检查函数定义
        if "def " not in failed_code:
            issues.append("缺少函数定义")
        
        # 检查常见错误模式
        if error_category in ["shape_error", "runtime_error"]:
            if "tl.load" in failed_code and "mask=" not in failed_code:
                issues.append("内存访问缺少mask保护")
            
            if "tl.program_id" not in failed_code:
                issues.append("缺少program_id获取")
        
        return issues
    
    def _assess_error_severity(self, error_category: str, error_message: str) -> str:
        """评估错误严重程度"""
        
        if error_category in ["import_error", "syntax_error"]:
            return "high"
        elif error_category in ["cuda_error", "memory_error"]:
            return "critical"
        elif error_category in ["shape_error", "kernel_launch_error"]:
            return "medium"
        else:
            return "low"
    
    async def _generate_fix_suggestions(self, error_info: Dict[str, Any], failed_code: str,
                                      pytorch_code: str, root_cause_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成修复建议"""
        
        # 构造修复建议提示
        fix_prompt = self._create_fix_suggestions_prompt(
            error_info, failed_code, pytorch_code, root_cause_analysis
        )
        
        # 调用LLM生成修复建议
        llm_response = self.generate_llm_response(
            fix_prompt,
            "You are an expert Triton kernel debugger. Provide specific, actionable fix suggestions."
        )
        
        # 解析修复建议
        suggestions = self._parse_fix_suggestions(llm_response, root_cause_analysis)
        
        return suggestions
    
    def _create_fix_suggestions_prompt(self, error_info: Dict[str, Any], failed_code: str,
                                     pytorch_code: str, root_cause_analysis: Dict[str, Any]) -> str:
        """创建修复建议提示"""
        
        error_message = error_info.get("error_message", "")
        error_type = error_info.get("error_type", "unknown")
        primary_cause = root_cause_analysis.get("primary_cause", "unknown")
        
        prompt = f"""
分析以下Triton kernel错误并提供具体的修复建议：

## 错误信息：
- 错误类型: {error_type}
- 错误消息: {error_message}
- 主要原因: {primary_cause}

## 有问题的代码：
```python
{failed_code}
```

## PyTorch参考代码：
```python
{pytorch_code}
```

## 根因分析：
{root_cause_analysis}

请提供：
1. 具体的修复步骤
2. 代码修改建议
3. 修复的优先级
4. 预防类似错误的方法
5. 设计层面的改进建议

提供详细的修复指导：
"""
        
        return prompt
    
    def _parse_fix_suggestions(self, llm_response: str, root_cause_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """解析修复建议"""
        
        fix_strategy = root_cause_analysis.get("fix_strategy", "comprehensive_review")
        
        suggestions = {
            "fix_strategy": fix_strategy,
            "priority_fixes": self._extract_priority_fixes(llm_response),
            "code_modifications": self._extract_code_modifications(llm_response),
            "design_improvements": self._extract_design_improvements(llm_response),
            "prevention_tips": self._extract_prevention_tips(llm_response),
            "detailed_guidance": llm_response
        }
        
        return suggestions
    
    def _extract_priority_fixes(self, llm_response: str) -> list:
        """提取优先修复项"""
        
        fixes = []
        
        if "导入" in llm_response or "import" in llm_response.lower():
            fixes.append("修复导入语句")
        if "语法" in llm_response or "syntax" in llm_response.lower():
            fixes.append("修复语法错误")
        if "装饰器" in llm_response or "decorator" in llm_response.lower():
            fixes.append("修复装饰器问题")
        if "形状" in llm_response or "shape" in llm_response.lower():
            fixes.append("修复张量形状问题")
        
        return fixes if fixes else ["根据详细指导进行修复"]
    
    def _extract_code_modifications(self, llm_response: str) -> list:
        """提取代码修改建议"""
        
        modifications = []
        
        if "@triton.jit" in llm_response:
            modifications.append("添加或修复@triton.jit装饰器")
        if "mask" in llm_response:
            modifications.append("添加内存访问mask保护")
        if "tl.constexpr" in llm_response:
            modifications.append("使用tl.constexpr标记常量")
        
        return modifications if modifications else ["参考详细指导进行修改"]
    
    def _extract_design_improvements(self, llm_response: str) -> list:
        """提取设计改进建议"""
        
        improvements = []
        
        if "重新设计" in llm_response:
            improvements.append("重新设计整体架构")
        if "算法" in llm_response:
            improvements.append("改进算法逻辑")
        if "内存" in llm_response:
            improvements.append("优化内存访问设计")
        
        return improvements if improvements else ["根据错误类型进行相应设计调整"]
    
    def _extract_prevention_tips(self, llm_response: str) -> list:
        """提取预防建议"""
        
        tips = []
        
        if "测试" in llm_response or "test" in llm_response.lower():
            tips.append("增加单元测试")
        if "验证" in llm_response or "validate" in llm_response.lower():
            tips.append("加强输入验证")
        if "检查" in llm_response or "check" in llm_response.lower():
            tips.append("添加运行时检查")
        
        return tips if tips else ["遵循Triton编程最佳实践"]
