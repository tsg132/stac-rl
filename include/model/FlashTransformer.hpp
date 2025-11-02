#pragma once

#include "common/Types.hpp"
#include <vector>
#include <memory>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace stac::model {

// ============================================================================
// FlashAttention Configuration
// ============================================================================

struct FlashAttentionConfig {
    int block_size_q = 64;      // Block size for queries
    int block_size_k = 64;      // Block size for keys  
    float softmax_scale = -1.0f; // Auto-compute if negative
    bool causal_mask = false;    // For autoregressive models
    float dropout_p = 0.0f;      // Attention dropout
    bool use_bias = false;       // Attention bias
};

// ============================================================================
// Linear Layer with Optimized GEMM
// ============================================================================

class LinearFlash {
public:
    LinearFlash(int in_features, int out_features, bool bias = true);
    ~LinearFlash();
    
    /**
     * Forward pass with optional activation fusion
     * @param input [batch_size * seq_len, in_features]
     * @param output [batch_size * seq_len, out_features]
     */
    void forward(const float* input, float* output, int batch_size) const;
    
#ifdef USE_CUDA
    /**
     * CUDA forward with cuBLAS/CUTLASS
     */
    void forward_cuda(const float* input, float* output, int batch_size) const;
#endif
    
    // Weight access
    float* weights() { return weights_; }
    float* bias() { return bias_; }
    const float* weights() const { return weights_; }
    const float* bias() const { return bias_; }
    
    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }
    
    // Memory management
    void to_device(int device_id);
    void to_cpu();
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    
    // CPU storage
    float* weights_;     // [out_features, in_features] 
    float* bias_;        // [out_features]
    
#ifdef USE_CUDA
    // GPU storage
    float* d_weights_;
    float* d_bias_;
    int device_id_;
#endif
    
    void initialize_weights();
    void allocate_memory();
    void free_memory();
};

// ============================================================================
// RMSNorm (More efficient than LayerNorm)
// ============================================================================

class RMSNorm {
public:
    explicit RMSNorm(int dim, float eps = 1e-5f);
    ~RMSNorm();
    
    /**
     * Forward pass: y = x / RMS(x) * gamma
     */
    void forward(const float* input, float* output, int batch_size) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* input, float* output, int batch_size) const;
#endif
    
    float* gamma() { return gamma_; }
    const float* gamma() const { return gamma_; }
    
    void to_device(int device_id);
    void to_cpu();
    
private:
    int dim_;
    float eps_;
    float* gamma_;
    
#ifdef USE_CUDA
    float* d_gamma_;
    int device_id_;
#endif
};

// ============================================================================
// FlashAttention Implementation
// ============================================================================

class FlashAttention {
public:
    FlashAttention(int dim, int num_heads, const FlashAttentionConfig& config = {});
    ~FlashAttention();
    
    /**
     * Forward pass with FlashAttention algorithm
     * @param qkv [batch_size, seq_len, 3 * dim] - packed QKV
     * @param output [batch_size, seq_len, dim]
     * @param attn_mask Optional mask [batch_size, seq_len, seq_len]
     */
    void forward(const float* qkv, 
                float* output,
                int batch_size,
                int seq_len,
                const float* attn_mask = nullptr) const;
    
#ifdef USE_CUDA
    /**
     * CUDA FlashAttention kernel
     * Uses tiling to reduce HBM accesses
     */
    void forward_cuda(const float* qkv,
                     float* output, 
                     int batch_size,
                     int seq_len,
                     const float* attn_mask = nullptr) const;
#endif
    
    // CPU implementation with tiling
    void forward_cpu_tiled(const float* qkv,
                          float* output,
                          int batch_size, 
                          int seq_len) const;
    
    int dim() const { return dim_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }
    
private:
    int dim_;
    int num_heads_;
    int head_dim_;
    float scale_;
    FlashAttentionConfig config_;
    
    // Internal buffers for tiled computation
    mutable std::vector<float> m_buffer_;     // Row maxes
    mutable std::vector<float> l_buffer_;     // Row sums  
    mutable std::vector<float> o_buffer_;     // Output accumulator
    
#ifdef USE_CUDA
    // CUDA workspace
    mutable float* d_workspace_;
    size_t workspace_size_;
#endif
    
    // Helper functions for block-wise computation
    void compute_block_attention(
        const float* q_block,
        const float* k_block, 
        const float* v_block,
        float* o_block,
        float* m_new,
        float* l_new,
        int block_size_q,
        int block_size_k,
        int head_dim) const;
};

// ============================================================================
// Fused MHA Layer (QKV projection + FlashAttention + Output projection)
// ============================================================================

class MultiHeadFlashAttention {
public:
    MultiHeadFlashAttention(int dim, int num_heads, 
                           const FlashAttentionConfig& config = {});
    ~MultiHeadFlashAttention();
    
    /**
     * Forward pass with fused operations
     * @param input [batch_size, seq_len, dim]
     * @param output [batch_size, seq_len, dim]
     */
    void forward(const float* input,
                float* output,
                int batch_size,
                int seq_len) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* input,
                     float* output,
                     int batch_size,
                     int seq_len) const;
#endif
    
    // Component access  
    LinearFlash& qkv_proj() { return *qkv_proj_; }
    LinearFlash& out_proj() { return *out_proj_; }
    FlashAttention& attention() { return *attention_; }
    
    void to_device(int device_id);
    void to_cpu();
    
    int num_parameters() const;
    
private:
    int dim_;
    int num_heads_;
    
    std::unique_ptr<LinearFlash> qkv_proj_;      // [dim, 3*dim]
    std::unique_ptr<FlashAttention> attention_;
    std::unique_ptr<LinearFlash> out_proj_;      // [dim, dim]
    
    // Buffers
    mutable std::vector<float> qkv_buffer_;
};

// ============================================================================
// SwiGLU Activation (Better than GELU for transformers)
// ============================================================================

class SwiGLU {
public:
    explicit SwiGLU(int dim, int hidden_dim);
    ~SwiGLU();
    
    /**
     * SwiGLU(x) = (xW1 * σ(xW3)) W2
     * where σ is SiLU/Swish activation
     */
    void forward(const float* input, float* output, 
                int batch_size, int seq_len) const;
    
#ifdef USE_CUDA  
    void forward_cuda(const float* input, float* output,
                     int batch_size, int seq_len) const;
#endif
    
    LinearFlash& gate_proj() { return *gate_proj_; }
    LinearFlash& up_proj() { return *up_proj_; }
    LinearFlash& down_proj() { return *down_proj_; }
    
    void to_device(int device_id);
    void to_cpu();
    
    int num_parameters() const;
    
private:
    int dim_;
    int hidden_dim_;
    
    std::unique_ptr<LinearFlash> gate_proj_;   // [dim, hidden_dim]
    std::unique_ptr<LinearFlash> up_proj_;     // [dim, hidden_dim]  
    std::unique_ptr<LinearFlash> down_proj_;   // [hidden_dim, dim]
    
    // Activation function (SiLU/Swish)
    static float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }
    
    mutable std::vector<float> gate_buffer_;
    mutable std::vector<float> up_buffer_;
};

// ============================================================================
// Optimized Transformer Block
// ============================================================================

class FlashTransformerBlock {
public:
    FlashTransformerBlock(int dim, int num_heads, int hidden_dim,
                         const FlashAttentionConfig& config = {});
    ~FlashTransformerBlock();
    
    /**
     * Forward pass with pre-norm architecture
     * x = x + attn(norm1(x))
     * x = x + ffn(norm2(x))
     */
    void forward(const float* input, float* output,
                int batch_size, int seq_len) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* input, float* output,
                     int batch_size, int seq_len) const;
#endif
    
    // Components
    RMSNorm& norm1() { return *norm1_; }
    RMSNorm& norm2() { return *norm2_; }
    MultiHeadFlashAttention& attention() { return *attention_; }
    SwiGLU& ffn() { return *ffn_; }
    
    void to_device(int device_id);
    void to_cpu();
    
    int num_parameters() const;
    
private:
    int dim_;
    
    std::unique_ptr<RMSNorm> norm1_;
    std::unique_ptr<MultiHeadFlashAttention> attention_;
    std::unique_ptr<RMSNorm> norm2_;
    std::unique_ptr<SwiGLU> ffn_;
    
    // Residual buffers
    mutable std::vector<float> residual_;
    mutable std::vector<float> norm_buffer_;
    mutable std::vector<float> attn_output_;
};



// ============================================================================
// Square Flash Transformer
// ============================================================================

/**
 * High-performance transformer for chess with FlashAttention.
 * Optimized for 64-token sequences (8x8 board).
 */
class SquareFlashTransformer {
public:
    SquareFlashTransformer(int num_layers, int dim, int num_heads,
                          int hidden_dim, const FlashAttentionConfig& config = {});
    ~SquareFlashTransformer();
    
    /**
     * Forward pass through all layers
     * @param input [batch_size, 64, dim]
     * @param output [batch_size, 64, dim]
     */
    void forward(const float* input, float* output, int batch_size) const;
    
#ifdef USE_CUDA
    void forward_cuda(const float* input, float* output, int batch_size) const;
#endif
    
    // Layer access
    FlashTransformerBlock& layer(int idx) { return *layers_[idx]; }
    const FlashTransformerBlock& layer(int idx) const { return *layers_[idx]; }
    RMSNorm& final_norm() { return *final_norm_; }
    
    int num_layers() const { return layers_.size(); }
    int dim() const { return dim_; }
    int num_parameters() const;
    
    void to_device(int device_id);
    void to_cpu();
    
    // Optimizations
    void enable_gradient_checkpointing(bool enable = true) {
        gradient_checkpointing_ = enable;
    }
    
    void set_inference_mode(bool inference = true) {
        inference_mode_ = inference;
    }
    
private:
    int dim_;
    int num_layers_;
    bool gradient_checkpointing_;
    bool inference_mode_;
    
    std::vector<std::unique_ptr<FlashTransformerBlock>> layers_;
    std::unique_ptr<RMSNorm> final_norm_;
    
    // Layer buffers for ping-pong
    mutable std::vector<float> buffer1_;
    mutable std::vector<float> buffer2_;
    
#ifdef USE_CUDA
    mutable float* d_buffer1_;
    mutable float* d_buffer2_;
#endif
};

// ============================================================================
// Rotary Position Embedding (RoPE) - Optional enhancement
// ============================================================================

class RotaryEmbedding {
public:
    RotaryEmbedding(int dim, int max_seq_len = 64, float base = 10000.0f);
    ~RotaryEmbedding();
    
    /**
     * Apply rotary embeddings to Q and K
     */
    void apply(float* q, float* k, int batch_size, int seq_len, int num_heads) const;
    
#ifdef USE_CUDA
    void apply_cuda(float* q, float* k, int batch_size, int seq_len, int num_heads) const;
#endif
    
private:
    int dim_;
    int max_seq_len_;
    float base_;
    
    std::vector<float> cos_cached_;
    std::vector<float> sin_cached_;
    
#ifdef USE_CUDA
    float* d_cos_cached_;
    float* d_sin_cached_;
#endif
    
    void compute_cache();
};

} // namespace stac::model
