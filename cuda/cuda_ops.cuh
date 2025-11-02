#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace stac::cuda {

// Activation functions
void silu_forward(const float* input, float* output, int n, cudaStream_t stream = 0);
void rmsnorm_forward(const float* input, const float* gamma, float* output,
                    int batch_size, int dim, float eps, cudaStream_t stream = 0);

// Loss computations
void compute_ppo_policy_loss(const float* log_probs, const float* old_log_probs,
                            const float* advantages, float clip_epsilon,
                            float* loss, int n, cudaStream_t stream = 0);

void compute_value_loss(const float* values, const float* returns,
                       const float* old_values, float clip_epsilon,
                       float* loss, int n, cudaStream_t stream = 0);

// Gradient operations
void clip_grad_norm(float* grad, float max_norm, int n, cudaStream_t stream = 0);
void zero_gradients(float* grad, int n, cudaStream_t stream = 0);

} // namespace stac::cuda
