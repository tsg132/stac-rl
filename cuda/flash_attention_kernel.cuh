#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

namespace stac::cuda {

// ============================================================================
// FlashAttention CUDA Kernel Configuration
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;

struct FlashAttentionParams {
    const float* Q;           // [batch * heads, seq_len, head_dim]
    const float* K;           // [batch * heads, seq_len, head_dim]
    const float* V;           // [batch * heads, seq_len, head_dim]
    float* O;                 // [batch * heads, seq_len, head_dim]
    float* M;                 // [batch * heads, seq_len] - row max
    float* L;                 // [batch * heads, seq_len] - row sum
    
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;
    bool causal_mask;
    
    int block_q;              // Query block size
    int block_k;              // Key/Value block size
};

// ============================================================================
// Forward Pass Kernel
// ============================================================================

/**
 * FlashAttention forward kernel
 * One thread block processes one attention head
 */
__global__ void flash_attention_forward_kernel(FlashAttentionParams params);

/**
 * Optimized kernel for small sequence lengths (N <= 64)
 */
__global__ void flash_attention_forward_small_kernel(FlashAttentionParams params);

// ============================================================================
// Backward Pass Kernels
// ============================================================================

struct FlashAttentionGradParams {
    // Forward inputs
    const float* Q;
    const float* K;
    const float* V;
    const float* O;
    const float* M;
    const float* L;
    
    // Gradients
    const float* dO;          // [batch * heads, seq_len, head_dim]
    float* dQ;                // [batch * heads, seq_len, head_dim]
    float* dK;                // [batch * heads, seq_len, head_dim]
    float* dV;                // [batch * heads, seq_len, head_dim]
    
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;
    bool causal_mask;
    
    int block_q;
    int block_k;
};

__global__ void flash_attention_backward_kernel(FlashAttentionGradParams params);

// ============================================================================
// Host Interface Functions
// ============================================================================

void launch_flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* M, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal_mask,
    cudaStream_t stream = 0);

void launch_flash_attention_backward(
    const float* Q, const float* K, const float* V,
    const float* O, const float* M, const float* L,
    const float* dO, float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal_mask,
    cudaStream_t stream = 0);

// ============================================================================
// Utility Functions
// ============================================================================

// Softmax with numerical stability
__device__ inline float safe_exp(float x, float max_val) {
    return expf(x - max_val);
}

// Warp-level reduction for max
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for max using shared memory
__device__ inline float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// Block-level reduction for sum using shared memory
__device__ inline float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

} // namespace stac::cuda
