#pragma once

#include "common/Types.hpp"
#include <vector>
#include <memory>

#ifdef USE_CUDA
// Forward declaration for CUDA types
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
#endif

namespace stac::training {

// ============================================================================
// Base Optimizer Interface
// ============================================================================

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /**
     * Perform single optimization step
     */
    virtual void step() = 0;
    
    /**
     * Zero all gradients
     */
    virtual void zero_grad() = 0;
    
    /**
     * Get/set learning rate
     */
    virtual float get_lr() const = 0;
    virtual void set_lr(float lr) = 0;
    
    /**
     * Add parameter group (weights + gradients)
     */
    virtual void add_param_group(float* params, float* grads, int size) = 0;
};

// ============================================================================
// Adam Optimizer
// ============================================================================

struct AdamConfig {
    float lr = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
    bool amsgrad = false;
};

class AdamOptimizer : public Optimizer {
public:
    explicit AdamOptimizer(const AdamConfig& config = {});
    ~AdamOptimizer() override;
    
    void step() override;
    void zero_grad() override;
    
    float get_lr() const override { return config_.lr; }
    void set_lr(float lr) override { config_.lr = lr; }
    
    void add_param_group(float* params, float* grads, int size) override;
    
#ifdef USE_CUDA
    void to_cuda(int device_id);
#endif
    
private:
    struct ParamGroup {
        float* params;
        float* grads;
        float* m;              // First moment estimate
        float* v;              // Second moment estimate
        int size;
        
#ifdef USE_CUDA
        float* d_params;
        float* d_grads;
        float* d_m;
        float* d_v;
#endif
    };
    
    AdamConfig config_;
    std::vector<ParamGroup> param_groups_;
    int step_count_;
    bool use_cuda_;
    int device_id_;
};

// ============================================================================
// SGD with Momentum
// ============================================================================

struct SGDConfig {
    float lr = 1e-3f;
    float momentum = 0.9f;
    float weight_decay = 0.0f;
    bool nesterov = false;
};

class SGDOptimizer : public Optimizer {
public:
    explicit SGDOptimizer(const SGDConfig& config = {});
    ~SGDOptimizer() override;
    
    void step() override;
    void zero_grad() override;
    
    float get_lr() const override { return config_.lr; }
    void set_lr(float lr) override { config_.lr = lr; }
    
    void add_param_group(float* params, float* grads, int size) override;
    
private:
    struct ParamGroup {
        float* params;
        float* grads;
        float* velocity;
        int size;
    };
    
    SGDConfig config_;
    std::vector<ParamGroup> param_groups_;
};

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

class LRScheduler {
public:
    explicit LRScheduler(Optimizer* optimizer) : optimizer_(optimizer) {}
    virtual ~LRScheduler() = default;
    
    virtual void step() = 0;
    virtual float get_lr() const { return optimizer_->get_lr(); }
    
protected:
    Optimizer* optimizer_;
};

class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer* optimizer, float gamma)
        : LRScheduler(optimizer), gamma_(gamma), base_lr_(optimizer->get_lr()) {}
    
    void step() override {
        float new_lr = optimizer_->get_lr() * gamma_;
        optimizer_->set_lr(new_lr);
    }
    
private:
    float gamma_;
    float base_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min = 0.0f)
        : LRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min),
          base_lr_(optimizer->get_lr()), step_count_(0) {}
    
    void step() override;
    
private:
    int T_max_;
    float eta_min_;
    float base_lr_;
    int step_count_;
};

} // namespace stac::training
