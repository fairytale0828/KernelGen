"""
代码验证服务 - 提供静态和动态验证
"""

import logging
import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    score: float  # 0-1之间的质量分数

class CodeValidationService:
    """代码验证服务"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
    
    def validate_triton_code(self, code: str) -> ValidationResult:
        """
        验证Triton代码
        
        Args:
            code: Triton代码字符串
            
        Returns:
            验证结果
        """
        issues = []
        warnings = []
        suggestions = []
        
        # 1. 基础语法检查
        syntax_issues = self._check_syntax(code)
        issues.extend(syntax_issues)
        
        # 2. Triton特定检查
        triton_issues, triton_warnings = self._check_triton_constraints(code)
        issues.extend(triton_issues)
        warnings.extend(triton_warnings)
        
        # 3. 代码结构检查
        structure_issues, structure_suggestions = self._check_code_structure(code)
        issues.extend(structure_issues)
        suggestions.extend(structure_suggestions)
        
        # 4. 性能相关检查
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            perf_warnings, perf_suggestions = self._check_performance_patterns(code)
            warnings.extend(perf_warnings)
            suggestions.extend(perf_suggestions)
        
        # 计算质量分数
        score = self._calculate_quality_score(len(issues), len(warnings), len(suggestions))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            score=score
        )
    
    def _check_syntax(self, code: str) -> List[str]:
        """检查Python语法"""
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"语法错误: {e.msg} (行 {e.lineno})")
        except Exception as e:
            issues.append(f"代码解析错误: {str(e)}")
        
        return issues
    
    def _check_triton_constraints(self, code: str) -> Tuple[List[str], List[str]]:
        """检查Triton特定约束"""
        issues = []
        warnings = []
        
        # 检查必要的导入
        if "import triton" not in code:
            issues.append("缺少必要的导入: import triton")
        if "import triton.language as tl" not in code:
            issues.append("缺少必要的导入: import triton.language as tl")
        
        # 检查@triton.jit装饰器
        if "@triton.jit" not in code:
            warnings.append("未找到@triton.jit装饰器")
        
        # 检查program_id使用
        program_id_pattern = r'tl\.program_id\(\s*axis\s*=\s*(\d+)\s*\)'
        matches = re.findall(program_id_pattern, code)
        for match in matches:
            axis = int(match)
            if axis > 2:
                issues.append(f"program_id axis超出限制: axis={axis} (最大支持2)")
        
        # 检查tl.load和tl.store的使用
        if "tl.load" in code and "mask=" not in code:
            warnings.append("tl.load调用可能缺少mask参数")
        
        if "tl.store" in code and "mask=" not in code:
            warnings.append("tl.store调用可能缺少mask参数")
        
        return issues, warnings
    
    def _check_code_structure(self, code: str) -> Tuple[List[str], List[str]]:
        """检查代码结构"""
        issues = []
        suggestions = []
        
        # 检查是否包含kernel函数和wrapper函数
        if not re.search(r'def\s+\w+.*\(.*\):', code):
            issues.append("代码中未找到函数定义")
        
        # 检查wrapper函数
        wrapper_pattern = r'def\s+(\w+)\s*\([^)]*torch\.Tensor[^)]*\)'
        if not re.search(wrapper_pattern, code):
            suggestions.append("建议添加接受torch.Tensor参数的wrapper函数")
        
        # 检查返回语句
        if "return" not in code:
            suggestions.append("wrapper函数应该包含返回语句")
        
        # 检查CUDA断言
        if "assert" in code and "cuda" in code.lower():
            suggestions.append("建议添加CUDA设备检查")
        
        return issues, suggestions
    
    def _check_performance_patterns(self, code: str) -> Tuple[List[str], List[str]]:
        """检查性能相关模式"""
        warnings = []
        suggestions = []
        
        # 检查BLOCK_SIZE使用
        if "BLOCK_SIZE" in code:
            # 检查是否使用了constexpr
            if "BLOCK_SIZE: tl.constexpr" not in code:
                warnings.append("BLOCK_SIZE应该声明为tl.constexpr")
        
        # 检查内存访问模式
        if "tl.load" in code:
            # 建议使用连续内存访问
            suggestions.append("确保内存访问模式是连续的以提高性能")
        
        # 检查循环嵌套
        loop_count = code.count("for ")
        if loop_count > 3:
            warnings.append(f"检测到{loop_count}个循环，可能影响性能")
        
        return warnings, suggestions
    
    def _calculate_quality_score(self, num_issues: int, num_warnings: int, num_suggestions: int) -> float:
        """计算代码质量分数"""
        # 基础分数
        base_score = 1.0
        
        # 错误扣分
        base_score -= num_issues * 0.3
        
        # 警告扣分
        base_score -= num_warnings * 0.1
        
        # 建议不扣分，但影响最终评级
        
        return max(0.0, min(1.0, base_score))
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """获取验证摘要"""
        summary = f"代码质量分数: {result.score:.2f}\n"
        
        if result.is_valid:
            summary += "✅ 代码验证通过\n"
        else:
            summary += "❌ 代码验证失败\n"
        
        if result.issues:
            summary += f"\n🚨 发现 {len(result.issues)} 个问题:\n"
            for issue in result.issues:
                summary += f"  - {issue}\n"
        
        if result.warnings:
            summary += f"\n⚠️  发现 {len(result.warnings)} 个警告:\n"
            for warning in result.warnings:
                summary += f"  - {warning}\n"
        
        if result.suggestions:
            summary += f"\n💡 {len(result.suggestions)} 个改进建议:\n"
            for suggestion in result.suggestions:
                summary += f"  - {suggestion}\n"
        
        return summary