#include "training/Optimizer.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef USE_CUDA
extern "C" {
    // External declarations for CUDA functions (implemented in .cu files)
    void* cuda_malloc(size_t size);
    void cuda_free(void* ptr);
    void cuda_memcpy_h2d(void* dst, const void* src, size_t size);
    void cuda_memset(void* ptr, int value, size_t size);
    void cuda_set_device(int device);
}
namespace stac::cuda {
    void zero_gradients(float* grads, int size);
}
#endif

namespace stac::training {

// ============================================================================
// Adam Optimizer Implementation
// ============================================================================

AdamOptimizer::AdamOptimizer(const AdamConfig& config)
    : config_(config), step_count_(0), use_cuda_(false), device_id_(-1) {
}

AdamOptimizer::~AdamOptimizer() {
    for (auto& group : param_groups_) {
        delete[] group.m;
        delete[] group.v;
        
#ifdef USE_CUDA
        if (use_cuda_) {
            cuda_free(group.d_m);
            cuda_free(group.d_v);
        }
#endif
    }
}

void AdamOptimizer::add_param_group(float* params, float* grads, int size) {
    ParamGroup group;
    group.params = params;
    group.grads = grads;
    group.size = size;
    
    // Allocate moment buffers
    group.m = new float[size];
    group.v = new float[size];
    std::fill(group.m, group.m + size, 0.0f);
    std::fill(group.v, group.v + size, 0.0f);
    
    group.d_params = nullptr;
    group.d_grads = nullptr;
    group.d_m = nullptr;
    group.d_v = nullptr;
    
#ifdef USE_CUDA
    if (use_cuda_) {
        group.d_params = (float*)cuda_malloc(size * sizeof(float));
        group.d_grads = (float*)cuda_malloc(size * sizeof(float));
        group.d_m = (float*)cuda_malloc(size * sizeof(float));
        group.d_v = (float*)cuda_malloc(size * sizeof(float));
        
        cuda_memset(group.d_m, 0, size * sizeof(float));
        cuda_memset(group.d_v, 0, size * sizeof(float));
    }
#endif
    
    param_groups_.push_back(group);
}

void AdamOptimizer::step() {
    step_count_++;
    
    float bias_correction1 = 1.0f - std::pow(config_.beta1, step_count_);
    float bias_correction2 = 1.0f - std::pow(config_.beta2, step_count_);
    float step_size = config_.lr * std::sqrt(bias_correction2) / bias_correction1;
    
    for (auto& group : param_groups_) {
#ifdef USE_CUDA
        if (use_cuda_) {
            // CUDA implementation (simplified)
            // In practice, this would call a custom CUDA kernel
            for (int i = 0; i < group.size; ++i) {
                float grad = group.grads[i];
                
                // Weight decay
                if (config_.weight_decay > 0) {
                    grad += config_.weight_decay * group.params[i];
                }
                
                // Update biased first moment estimate
                group.m[i] = config_.beta1 * group.m[i] + (1 - config_.beta1) * grad;
                
                // Update biased second raw moment estimate
                group.v[i] = config_.beta2 * group.v[i] + (1 - config_.beta2) * grad * grad;
                
                // Compute bias-corrected moment estimates
                float m_hat = group.m[i] / bias_correction1;
                float v_hat = group.v[i] / bias_correction2;
                
                // Update parameters
                group.params[i] -= step_size * m_hat / (std::sqrt(v_hat) + config_.eps);
            }
        } else
#endif
        {
            // CPU implementation
            for (int i = 0; i < group.size; ++i) {
                float grad = group.grads[i];
                
                // Weight decay
                if (config_.weight_decay > 0) {
                    grad += config_.weight_decay * group.params[i];
                }
                
                // Update biased first moment estimate
                group.m[i] = config_.beta1 * group.m[i] + (1 - config_.beta1) * grad;
                
                // Update biased second raw moment estimate
                group.v[i] = config_.beta2 * group.v[i] + (1 - config_.beta2) * grad * grad;
                
                // Compute bias-corrected moment estimates
                float m_hat = group.m[i] / bias_correction1;
                float v_hat = group.v[i] / bias_correction2;
                
                // Update parameters
                group.params[i] -= step_size * m_hat / (std::sqrt(v_hat) + config_.eps);
            }
        }
    }
}

void AdamOptimizer::zero_grad() {
    for (auto& group : param_groups_) {
#ifdef USE_CUDA
        if (use_cuda_) {
            cuda::zero_gradients(group.d_grads, group.size);
        } else
#endif
        {
            std::fill(group.grads, group.grads + group.size, 0.0f);
        }
    }
}

#ifdef USE_CUDA
void AdamOptimizer::to_cuda(int device_id) {
    use_cuda_ = true;
    device_id_ = device_id;
    cudaSetDevice(device_id);
    
    for (auto& group : param_groups_) {
        if (!group.d_params) {
            cudaMalloc(&group.d_params, group.size * sizeof(float));
            cudaMalloc(&group.d_grads, group.size * sizeof(float));
            cudaMalloc(&group.d_m, group.size * sizeof(float));
            cudaMalloc(&group.d_v, group.size * sizeof(float));
            
            cudaMemcpy(group.d_params, group.params, 
                      group.size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(group.d_m, 0, group.size * sizeof(float));
            cudaMemset(group.d_v, 0, group.size * sizeof(float));
        }
    }
}
#endif

// ============================================================================
// SGD Optimizer Implementation
// ============================================================================

SGDOptimizer::SGDOptimizer(const SGDConfig& config) : config_(config) {
}

SGDOptimizer::~SGDOptimizer() {
    for (auto& group : param_groups_) {
        delete[] group.velocity;
    }
}

void SGDOptimizer::add_param_group(float* params, float* grads, int size) {
    ParamGroup group;
    group.params = params;
    group.grads = grads;
    group.size = size;
    
    group.velocity = new float[size];
    std::fill(group.velocity, group.velocity + size, 0.0f);
    
    param_groups_.push_back(group);
}

void SGDOptimizer::step() {
    for (auto& group : param_groups_) {
        for (int i = 0; i < group.size; ++i) {
            float grad = group.grads[i];
            
            // Weight decay
            if (config_.weight_decay > 0) {
                grad += config_.weight_decay * group.params[i];
            }
            
            // Update velocity
            group.velocity[i] = config_.momentum * group.velocity[i] + grad;
            
            // Update parameters
            if (config_.nesterov) {
                group.params[i] -= config_.lr * (grad + config_.momentum * group.velocity[i]);
            } else {
                group.params[i] -= config_.lr * group.velocity[i];
            }
        }
    }
}

void SGDOptimizer::zero_grad() {
    for (auto& group : param_groups_) {
        std::fill(group.grads, group.grads + group.size, 0.0f);
    }
}

// ============================================================================
// Cosine Annealing LR Scheduler
// ============================================================================

void CosineAnnealingLR::step() {
    step_count_++;
    float cos_val = std::cos(M_PI * step_count_ / T_max_);
    float new_lr = eta_min_ + (base_lr_ - eta_min_) * (1.0f + cos_val) / 2.0f;
    optimizer_->set_lr(new_lr);
}

} // namespace stac::training
