# Generated Triton Kernel (LangChain Multi-Agent)
# Level: 1
# Problem ID: 19
# Operation: 19_ReLU
# Generated at: 2025-12-04 06:36:23
# Best Speedup: 1.00x

import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for ReLU activation.
    
    Args:
        x_ptr: Pointer to input tensor.
        output_ptr: Pointer to output tensor.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Number of elements processed by each thread block.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    # Using tl.where for branchless implementation
    result = tl.where(x > 0, x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def kernel_wrapper(x: torch.Tensor):
    """Wrapper function for ReLU kernel.
    
    Args:
        x: Input tensor of any shape.
        
    Returns:
        torch.Tensor: Output tensor with ReLU applied, same shape as input.
    """
    assert x.is_cuda, "Input must be on CUDA"
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Define grid function
    def grid(meta):
        return ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output