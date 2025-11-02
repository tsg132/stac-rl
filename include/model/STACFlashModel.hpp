#pragma once

#include "common/Types.hpp"
#include "common/Config.hpp"
#include "model/FlashTransformer.hpp"
#include "model/ActionSpace.hpp"
#include <memory>
#include <filesystem>

#ifdef USE_CUDA
// Forward declarations for CUDA types (avoid including headers in .hpp)
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
struct __half;
typedef struct __half half;
struct cublasContext;
typedef struct cublasContext* cublasHandle_t;
#endif

namespace stac::model {

// ============================================================================
// Optimized Observation Embedding
// ============================================================================

class FlashObservationEmbedding {
public:
    explicit FlashObservationEmbedding(int dim);
    ~FlashObservationEmbedding();
    
    /**
     * Embed observation planes to tokens with optimized layout
     * @param observation [batch_size, 18, 8, 8] in CHW format
     * @param tokens [batch_size, 64, dim]
     */
    void forward(const float* observation, float* tokens, int batch_size) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* observation, float* tokens, int batch_size) const;
#endif
    
    float* weights() { return weights_; }
    float* bias() { return bias_; }
    
    void to_device(int device_id);
    void to_cpu();
    
    int num_parameters() const { return 18 * dim_ + dim_; }
    
private:
    int dim_;
    float* weights_;  // [18, dim] - one weight per plane
    float* bias_;     // [dim]
    
#ifdef USE_CUDA
    float* d_weights_;
    float* d_bias_;
    int device_id_;
#endif
};

// ============================================================================
// Optimized Policy Head with Flash Attention
// ============================================================================

class FlashPolicyHead {
public:
    explicit FlashPolicyHead(int dim);
    ~FlashPolicyHead();
    
    /**
     * Generate policy logits using factorized attention
     * @param tokens [batch_size, 64, dim]
     * @param logits [batch_size, 4672]
     */
    void forward(const float* tokens, float* logits, int batch_size) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* tokens, float* logits, int batch_size) const;
#endif
    
    LinearFlash& from_proj() { return *from_proj_; }
    LinearFlash& plane_proj() { return *plane_proj_; }
    
    void to_device(int device_id);
    void to_cpu();
    
    int num_parameters() const;
    
private:
    int dim_;
    std::unique_ptr<LinearFlash> from_proj_;   // [dim, dim] per-square features
    std::unique_ptr<LinearFlash> plane_proj_;  // [dim, 73] plane logits
    std::unique_ptr<RMSNorm> norm_;           // Normalize before projection
};

// ============================================================================
// Optimized Value Head
// ============================================================================

class FlashValueHead {
public:
    explicit FlashValueHead(int dim);
    ~FlashValueHead();
    
    /**
     * Compute value with efficient pooling
     * @param tokens [batch_size, 64, dim]
     * @param values [batch_size]
     */
    void forward(const float* tokens, float* values, int batch_size) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* tokens, float* values, int batch_size) const;
#endif
    
    void to_device(int device_id);
    void to_cpu();
    
    int num_parameters() const;
    
private:
    int dim_;
    std::unique_ptr<RMSNorm> norm_;
    std::unique_ptr<LinearFlash> fc1_;  // [dim, dim]
    std::unique_ptr<LinearFlash> fc2_;  // [dim, 1]
    
    mutable std::vector<float> pooled_buffer_;
};

// ============================================================================
// STAC Model with FlashAttention
// ============================================================================

/**
 * High-performance Square-Transformer Actor-Critic model.
 * 
 * Features:
 * - FlashAttention for O(N) memory complexity
 * - SwiGLU activation for better gradient flow
 * - RMSNorm for faster normalization
 * - Optimized memory layout for GPU
 * - Mixed precision support (FP16/BF16)
 */
class STACFlashModel {
public:
    explicit STACFlashModel(const ModelConfig& config);
    ~STACFlashModel();
    
    // -------------------------------------------------------------------------
    // Forward Pass
    // -------------------------------------------------------------------------
    
    /**
     * Single position evaluation
     */
    ModelOutput forward(const ObservationTensor& observation) const;
    
    /**
     * Batch evaluation with optimizations
     */
    std::vector<ModelOutput> forward_batch(
        const std::vector<ObservationTensor>& observations) const;
    
    /**
     * Raw tensor interface for maximum performance
     */
    void forward_tensor(
        const float* observations,  // [batch_size, 18, 8, 8]
        float* policy_logits,       // [batch_size, 4672]
        float* values,              // [batch_size]
        int batch_size,
        bool use_cuda = true) const;
    
    // -------------------------------------------------------------------------
    // Optimized Inference
    // -------------------------------------------------------------------------
    
    /**
     * Stream-based async inference for multiple batches
     */
    void forward_async(
        const float* observations,
        float* policy_logits,
        float* values,
        int batch_size,
        cudaStream_t stream = 0) const;
    
    /**
     * Mixed precision inference (FP16)
     */
    void forward_fp16(
        const half* observations,
        half* policy_logits,
        half* values,
        int batch_size) const;
    
    // -------------------------------------------------------------------------
    // Model Management
    // -------------------------------------------------------------------------
    
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);
    void save_onnx(const std::filesystem::path& path) const;
    
    /**
     * Optimize model for inference (graph optimization, kernel fusion)
     */
    void optimize_for_inference();
    
    /**
     * Quantize model to INT8 for faster inference
     */
    void quantize_int8(const std::vector<ObservationTensor>& calibration_data);
    
    // -------------------------------------------------------------------------
    // Device Management
    // -------------------------------------------------------------------------
    
    void to_device(int device_id);
    void to_cpu();
    void to_multi_gpu(const std::vector<int>& device_ids);
    
    // -------------------------------------------------------------------------
    // Memory Management
    // -------------------------------------------------------------------------
    
    /**
     * Pre-allocate workspace for batch size
     */
    void allocate_workspace(int max_batch_size);
    
    /**
     * Get memory usage statistics
     */
    struct MemoryStats {
        size_t model_size;
        size_t workspace_size;
        size_t peak_memory;
    };
    MemoryStats get_memory_stats() const;
    
    // -------------------------------------------------------------------------
    // Components and Parameters
    // -------------------------------------------------------------------------
    
    FlashObservationEmbedding& embedding() { return *embedding_; }
    SquareFlashTransformer& transformer() { return *transformer_; }
    FlashPolicyHead& policy_head() { return *policy_head_; }
    FlashValueHead& value_head() { return *value_head_; }
    
    int num_parameters() const;
    std::vector<float*> get_parameters();
    
    // -------------------------------------------------------------------------
    // Training Support
    // -------------------------------------------------------------------------
    
    void train_mode(bool training = true);
    void eval_mode() { train_mode(false); }
    
    /**
     * Enable gradient checkpointing for memory-efficient training
     */
    void enable_gradient_checkpointing(bool enable = true);
    
    /**
     * Get parameter groups for optimizer
     */
    struct ParameterGroup {
        std::string name;
        std::vector<float*> params;
        float lr_scale;
        float weight_decay;
    };
    std::vector<ParameterGroup> get_parameter_groups() const;
    
    // -------------------------------------------------------------------------
    // Profiling and Debugging
    // -------------------------------------------------------------------------
    
    void enable_profiling(bool enable = true);
    void print_summary() const;
    void benchmark(int num_iterations = 100, int batch_size = 256);
    
private:
    ModelConfig config_;
    bool training_mode_;
    bool profiling_enabled_;
    int device_id_;
    
    // Model components
    std::unique_ptr<FlashObservationEmbedding> embedding_;
    std::unique_ptr<SquareFlashTransformer> transformer_;
    std::unique_ptr<FlashPolicyHead> policy_head_;
    std::unique_ptr<FlashValueHead> value_head_;
    
    // Workspace buffers
    mutable std::vector<float> cpu_workspace_;
    
#ifdef USE_CUDA
    // CUDA resources
    mutable float* d_tokens_;          // [max_batch, 64, dim]
    mutable float* d_transformer_out_; // [max_batch, 64, dim]
    mutable float* d_workspace_;       // General workspace
    size_t workspace_size_;
    int max_batch_size_;
    
    // CUDA streams for async execution
    cudaStream_t compute_stream_;
    cudaStream_t copy_stream_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
#endif
    
    // Private methods
    void allocate_buffers(int batch_size);
    void free_buffers();
};

// ============================================================================
// Model Factory with Presets
// ============================================================================

class FlashModelFactory {
public:
    /**
     * Create model from config
     */
    static std::unique_ptr<STACFlashModel> create(const ModelConfig& config);
    
    /**
     * Preset configurations
     */
    static std::unique_ptr<STACFlashModel> create_small();   // 4L, 256d, 8h
    static std::unique_ptr<STACFlashModel> create_base();    // 8L, 512d, 8h  
    static std::unique_ptr<STACFlashModel> create_large();   // 12L, 768d, 12h
    static std::unique_ptr<STACFlashModel> create_xlarge();  // 24L, 1024d, 16h
    
    /**
     * Load pretrained model
     */
    static std::unique_ptr<STACFlashModel> from_pretrained(
        const std::string& model_name,
        const std::filesystem::path& cache_dir = "~/.cache/stac");
};

// ============================================================================
// Model Ensemble for Stronger Play
// ============================================================================

class STACEnsemble {
public:
    explicit STACEnsemble(std::vector<std::unique_ptr<STACFlashModel>> models);
    
    /**
     * Forward with ensemble averaging
     */
    ModelOutput forward(const ObservationTensor& observation) const;
    
    /**
     * Forward with uncertainty estimation
     */
    struct EnsembleOutput {
        PolicyLogits mean_policy;
        float mean_value;
        PolicyLogits std_policy;
        float std_value;
    };
    EnsembleOutput forward_with_uncertainty(const ObservationTensor& observation) const;
    
private:
    std::vector<std::unique_ptr<STACFlashModel>> models_;
};

} // namespace stac::model
