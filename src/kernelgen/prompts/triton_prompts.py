"""
Triton代码生成提示构造器
基于KernelBench的提示模板
"""

def get_triton_generation_prompt(pytorch_code: str) -> str:
    """
    构造Triton代码生成提示
    
    Args:
        pytorch_code: PyTorch参考代码
        
    Returns:
        生成提示
    """
    
    prompt = f"""You are an expert in GPU kernel programming with Triton. Your task is to implement a high-performance Triton kernel that replicates the functionality of the given PyTorch model.

## Input PyTorch Model:
```python
{pytorch_code}
```

## Requirements:
1. **Functionality**: The Triton kernel must produce identical results to the PyTorch model
2. **Performance**: Optimize for GPU performance using Triton's block-level programming model
3. **Memory Efficiency**: Use efficient memory access patterns and minimize memory transfers
4. **Correctness**: Handle edge cases and ensure numerical stability

## Implementation Guidelines:
1. Use appropriate Triton decorators (@triton.jit)
2. Implement efficient block-level operations
3. Use proper memory coalescing patterns
4. Handle boundary conditions correctly
5. Include a wrapper function that can be called from Python

## Output Format:
Provide a complete Python file with:
1. All necessary imports (torch, triton, etc.)
2. The Triton kernel function with @triton.jit decorator
3. A wrapper function that handles tensor shapes and launches the kernel
4. The wrapper should have the same interface as the PyTorch model's forward method

## Example Structure:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    # input pointers
    # output pointers  
    # scalar parameters
    # block sizes
):
    # kernel implementation
    pass

def triton_wrapper(inputs):
    # tensor preparation
    # kernel launch
    # return results
    pass
```

Please implement the Triton kernel now:"""

    return prompt

def get_triton_optimization_prompt(kernel_code: str, performance_feedback: str) -> str:
    """
    构造Triton代码优化提示
    
    Args:
        kernel_code: 当前的Triton kernel代码
        performance_feedback: 性能反馈信息
        
    Returns:
        优化提示
    """
    
    prompt = f"""You are an expert in Triton GPU kernel optimization. Your task is to improve the performance of the given Triton kernel based on the performance feedback.

## Current Triton Kernel:
```python
{kernel_code}
```

## Performance Feedback:
{performance_feedback}

## Optimization Guidelines:
1. **Memory Access Optimization**:
   - Improve memory coalescing patterns
   - Reduce memory bank conflicts
   - Optimize shared memory usage

2. **Compute Optimization**:
   - Increase arithmetic intensity
   - Use vectorized operations where possible
   - Optimize block sizes for target GPU architecture

3. **Control Flow Optimization**:
   - Minimize divergent branches
   - Reduce loop overhead
   - Use efficient reduction patterns

4. **Triton-Specific Optimizations**:
   - Tune block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
   - Use appropriate data types
   - Leverage Triton's automatic optimization features

## Requirements:
1. Maintain functional correctness
2. Improve performance metrics (throughput, latency)
3. Keep the same interface
4. Add comments explaining optimizations

Please provide the optimized Triton kernel:"""

    return prompt

def get_triton_error_fix_prompt(kernel_code: str, error_message: str, suggestions) -> str:
    """
    构造错误修复提示
    
    Args:
        kernel_code: 有错误的kernel代码
        error_message: 错误信息
        suggestions: 修复建议
        
    Returns:
        错误修复提示
    """
    
    suggestions_text = "\n".join(f"- {s}" for s in suggestions)
    
    prompt = f"""You are an expert Triton GPU kernel programmer. The following kernel has errors that need to be fixed.

## Problematic Kernel Code:
```python
{kernel_code}
```

## Error Message:
{error_message}

## Fix Suggestions:
{suggestions_text}

## Requirements:
1. Fix all compilation and runtime errors
2. Maintain the original functionality
3. Ensure the kernel can be successfully compiled and executed
4. Provide complete, runnable code
5. Include necessary imports and wrapper functions

Please provide the corrected Triton kernel:"""

    return prompt

def get_triton_debug_prompt(kernel_code: str, error_message: str) -> str:
    """
    构造Triton代码调试提示
    
    Args:
        kernel_code: 有问题的Triton kernel代码
        error_message: 错误信息
        
    Returns:
        调试提示
    """
    
    prompt = f"""You are an expert in Triton GPU kernel debugging. Your task is to fix the errors in the given Triton kernel.

## Problematic Triton Kernel:
```python
{kernel_code}
```

## Error Message:
```
{error_message}
```

## Common Triton Issues to Check:
1. **Syntax Errors**:
   - Incorrect Triton language syntax
   - Missing or incorrect decorators
   - Invalid function signatures

2. **Memory Access Errors**:
   - Out-of-bounds memory access
   - Incorrect pointer arithmetic
   - Invalid tensor indexing

3. **Type Errors**:
   - Incompatible data types
   - Missing type annotations
   - Incorrect tensor dtypes

4. **Kernel Launch Errors**:
   - Incorrect grid/block configuration
   - Missing kernel arguments
   - Invalid tensor shapes

5. **Logic Errors**:
   - Incorrect algorithm implementation
   - Wrong boundary condition handling
   - Numerical instability

## Requirements:
1. Fix all compilation and runtime errors
2. Maintain the original functionality
3. Ensure the kernel can be successfully launched
4. Add error handling where appropriate

Please provide the corrected Triton kernel:"""

    return prompt