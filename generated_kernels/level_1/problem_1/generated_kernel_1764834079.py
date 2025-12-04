# Generated Triton Kernel (LangChain Multi-Agent)
# Level: 1
# Problem ID: 1
# Operation: 1_Square_matrix_multiplication_
# Generated at: 2025-12-04 07:41:19
# Best Speedup: 0.88x

import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_ASYNC_COPY: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A @ B
    
    Args:
        A_ptr: Pointer to matrix A of shape (M, K)
        B_ptr: Pointer to matrix B of shape (K, N)
        C_ptr: Pointer to output matrix C of shape (M, N)
        M, N, K: Matrix dimensions
        stride_*: Strides for each matrix dimension
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes
        GROUP_SIZE_M: Group size for load balancing
        USE_ASYNC_COPY: Whether to use asynchronous copy
    """
    
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets for the tile
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A tile
        a_mask = (ram[:, None] < M) & (rk[None, :] < K)
        a = tl.load(
            A_ptr + ram[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        
        # Load B tile
        b_mask = (rk[:, None] < K) & (rbn[None, :] < N)
        b = tl.load(
            B_ptr + rk[:, None] * stride_bk + rbn[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
        )
        
        # Matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store result
    c_mask = (ram[:, None] < M) & (rbn[None, :] < N)
    tl.store(
        C_ptr + ram[:, None] * stride_cm + rbn[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


def kernel_wrapper(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for matrix multiplication kernel
    
    Args:
        A: Input matrix A of shape (M, K)
        B: Input matrix B of shape (K, N)
        
    Returns:
        C: Output matrix C of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D matrices"
    assert A.shape[1] == B.shape[0], "Matrix dimensions must match for multiplication"
    
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    
    # Get matrix dimensions
    M, K = A.shape
    _, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    USE_ASYNC_COPY = True
    
    # Grid configuration
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Launch kernel
    kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        USE_ASYNC_COPY=USE_ASYNC_COPY,
    )
    
    return C