#include "cuda_ops.cuh"
#include <cmath>

namespace stac::cuda {

// ============================================================================
// Activation Kernels
// ============================================================================

__global__ void silu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void rmsnorm_forward_kernel(const float* input, const float* gamma,
                                       float* output, int batch_size, int dim, float eps) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * dim;
    float* y = output + batch_idx * dim;
    
    // Compute RMS
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum_sq += x[i] * x[i];
    }
    
    // Reduce across block
    __shared__ float shared[256];
    shared[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float rms = sqrtf(shared[0] / dim + eps);
    float scale = 1.0f / rms;
    
    // Apply normalization
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        y[i] = x[i] * scale * gamma[i];
    }
}

// ============================================================================
// Loss Kernels
// ============================================================================

__global__ void ppo_policy_loss_kernel(const float* log_probs, const float* old_log_probs,
                                       const float* advantages, float clip_epsilon,
                                       float* loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ratio = expf(log_probs[idx] - old_log_probs[idx]);
        float adv = advantages[idx];
        
        float surr1 = ratio * adv;
        float surr2 = fminf(fmaxf(ratio, 1.0f - clip_epsilon), 1.0f + clip_epsilon) * adv;
        
        loss[idx] = -fminf(surr1, surr2);
    }
}

__global__ void value_loss_kernel(const float* values, const float* returns,
                                 const float* old_values, float clip_epsilon,
                                 float* loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = values[idx];
        float v_old = old_values[idx];
        float ret = returns[idx];
        
        // Clipped value loss
        float v_clipped = v_old + fminf(fmaxf(v - v_old, -clip_epsilon), clip_epsilon);
        float loss1 = (v - ret) * (v - ret);
        float loss2 = (v_clipped - ret) * (v_clipped - ret);
        
        loss[idx] = fmaxf(loss1, loss2);
    }
}

// ============================================================================
// Gradient Operations
// ============================================================================

__global__ void clip_grad_norm_kernel(float* grad, float max_norm, float total_norm, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && total_norm > max_norm) {
        grad[idx] *= max_norm / (total_norm + 1e-6f);
    }
}

__global__ void compute_grad_norm_kernel(const float* grad, float* partial_sums, int n) {
    __shared__ float shared[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (idx < n) {
        sum = grad[idx] * grad[idx];
    }
    
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

__global__ void zero_grad_kernel(float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = 0.0f;
    }
}

// ============================================================================
// Host Interface Functions
// ============================================================================

void silu_forward(const float* input, float* output, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

void rmsnorm_forward(const float* input, const float* gamma, float* output,
                    int batch_size, int dim, float eps, cudaStream_t stream) {
    int blocks = batch_size;
    int threads = min(dim, 256);
    rmsnorm_forward_kernel<<<blocks, threads, 0, stream>>>(input, gamma, output, 
                                                           batch_size, dim, eps);
}

void compute_ppo_policy_loss(const float* log_probs, const float* old_log_probs,
                            const float* advantages, float clip_epsilon,
                            float* loss, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ppo_policy_loss_kernel<<<blocks, threads, 0, stream>>>(log_probs, old_log_probs,
                                                           advantages, clip_epsilon, loss, n);
}

void compute_value_loss(const float* values, const float* returns,
                       const float* old_values, float clip_epsilon,
                       float* loss, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    value_loss_kernel<<<blocks, threads, 0, stream>>>(values, returns, old_values,
                                                      clip_epsilon, loss, n);
}

void clip_grad_norm(float* grad, float max_norm, int n, cudaStream_t stream) {
    // First compute total norm
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));
    
    compute_grad_norm_kernel<<<blocks, threads, 0, stream>>>(grad, d_partial_sums, n);
    
    // Sum partial sums on CPU (simplified - should use reduction kernel)
    float* h_partial_sums = new float[blocks];
    cudaMemcpyAsync(h_partial_sums, d_partial_sums, blocks * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    float total_norm = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        total_norm += h_partial_sums[i];
    }
    total_norm = sqrtf(total_norm);
    
    // Clip gradients
    clip_grad_norm_kernel<<<blocks, threads, 0, stream>>>(grad, max_norm, total_norm, n);
    
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);
}

void zero_gradients(float* grad, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    zero_grad_kernel<<<blocks, threads, 0, stream>>>(grad, n);
}

} // namespace stac::cuda
