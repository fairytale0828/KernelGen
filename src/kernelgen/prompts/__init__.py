"""
提示模块
"""

from .triton_prompts import (
    get_triton_generation_prompt,
    get_triton_optimization_prompt,
    get_triton_debug_prompt,
    get_triton_error_fix_prompt
)

__all__ = [
    "get_triton_generation_prompt",
    "get_triton_optimization_prompt", 
    "get_triton_debug_prompt",
    "get_triton_error_fix_prompt"
]