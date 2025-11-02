#include "model/FlashTransformer.hpp"
#include <cstring>
#include <algorithm>
#include <random>
#include <cassert>
#include <immintrin.h>  // For SIMD

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace stac::model {

// ============================================================================
// LinearFlash Implementation
// ============================================================================

LinearFlash::LinearFlash(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
#ifdef USE_CUDA
    device_id_ = -1;
    d_weights_ = nullptr;
    d_bias_ = nullptr;
#endif
    allocate_memory();
    initialize_weights();
}

LinearFlash::~LinearFlash() {
    free_memory();
}

void LinearFlash::allocate_memory() {
    // Align to 64 bytes for better cache performance
    size_t weight_size = out_features_ * in_features_ * sizeof(float);
    weights_ = static_cast<float*>(std::aligned_alloc(64, weight_size));
    
    if (use_bias_) {
        size_t bias_size = out_features_ * sizeof(float);
        bias_ = static_cast<float*>(std::aligned_alloc(64, bias_size));
    } else {
        bias_ = nullptr;
    }
}

void LinearFlash::free_memory() {
    if (weights_) std::free(weights_);
    if (bias_) std::free(bias_);
    
#ifdef USE_CUDA
    if (d_weights_) cudaFree(d_weights_);
    if (d_bias_) cudaFree(d_bias_);
#endif
}

void LinearFlash::initialize_weights() {
    std::mt19937 rng(42);
    float scale = std::sqrt(2.0f / in_features_);
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < out_features_ * in_features_; ++i) {
        weights_[i] = dist(rng);
    }
    
    if (use_bias_) {
        std::fill(bias_, bias_ + out_features_, 0.0f);
    }
}

void LinearFlash::forward(const float* input, float* output, int batch_size) const {
    // Optimized GEMM with tiling for cache efficiency
    const int TILE_SIZE = 64;
    
    // Initialize output with bias
    if (use_bias_) {
        for (int b = 0; b < batch_size; ++b) {
            std::copy(bias_, bias_ + out_features_, output + b * out_features_);
        }
    } else {
        std::fill(output, output + batch_size * out_features_, 0.0f);
    }
    
    // Tiled matrix multiplication
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int o_tile = 0; o_tile < out_features_; o_tile += TILE_SIZE) {
            int o_end = std::min(o_tile + TILE_SIZE, out_features_);
            
            for (int i_tile = 0; i_tile < in_features_; i_tile += TILE_SIZE) {
                int i_end = std::min(i_tile + TILE_SIZE, in_features_);
                
                // Compute tile
                for (int o = o_tile; o < o_end; ++o) {
                    float sum = output[b * out_features_ + o];
                    
                    // Vectorized inner loop
                    #pragma omp simd reduction(+:sum)
                    for (int i = i_tile; i < i_end; ++i) {
                        sum += input[b * in_features_ + i] * weights_[o * in_features_ + i];
                    }
                    
                    output[b * out_features_ + o] = sum;
                }
            }
        }
    }
}

// ============================================================================
// RMSNorm Implementation
// ============================================================================

RMSNorm::RMSNorm(int dim, float eps)
    : dim_(dim), eps_(eps) {
    gamma_ = static_cast<float*>(std::aligned_alloc(64, dim_ * sizeof(float)));
    std::fill(gamma_, gamma_ + dim_, 1.0f);
    
#ifdef USE_CUDA
    device_id_ = -1;
    d_gamma_ = nullptr;
#endif
}

RMSNorm::~RMSNorm() {
    if (gamma_) std::free(gamma_);
#ifdef USE_CUDA
    if (d_gamma_) cudaFree(d_gamma_);
#endif
}

void RMSNorm::forward(const float* input, float* output, int batch_size) const {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* x = input + b * dim_;
        float* y = output + b * dim_;
        
        // Compute RMS
        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for (int i = 0; i < dim_; ++i) {
            sum_sq += x[i] * x[i];
        }
        
        float rms = std::sqrt(sum_sq / dim_ + eps_);
        float scale = 1.0f / rms;
        
        // Apply normalization and scaling
        #pragma omp simd
        for (int i = 0; i < dim_; ++i) {
            y[i] = x[i] * scale * gamma_[i];
        }
    }
}

// ============================================================================
// FlashAttention Implementation
// ============================================================================

FlashAttention::FlashAttention(int dim, int num_heads, const FlashAttentionConfig& config)
    : dim_(dim), num_heads_(num_heads), head_dim_(dim / num_heads), config_(config) {
    
    assert(dim_ % num_heads_ == 0);
    
    if (config_.softmax_scale < 0) {
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    } else {
        scale_ = config_.softmax_scale;
    }
    
#ifdef USE_CUDA
    d_workspace_ = nullptr;
    workspace_size_ = 0;
#endif
}

FlashAttention::~FlashAttention() {
#ifdef USE_CUDA
    if (d_workspace_) cudaFree(d_workspace_);
#endif
}

void FlashAttention::forward(const float* qkv, float* output,
                            int batch_size, int seq_len, 
                            const float* attn_mask) const {
    // Use CPU tiled implementation
    forward_cpu_tiled(qkv, output, batch_size, seq_len);
}

void FlashAttention::forward_cpu_tiled(const float* qkv, float* output,
                                       int batch_size, int seq_len) const {
    const int total_heads = batch_size * num_heads_;
    const int qkv_stride = 3 * dim_;
    
    // Ensure buffers are large enough
    if (m_buffer_.size() < total_heads * seq_len) {
        m_buffer_.resize(total_heads * seq_len);
        l_buffer_.resize(total_heads * seq_len);
        o_buffer_.resize(batch_size * seq_len * dim_);
    }
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            const int head_idx = b * num_heads_ + h;
            
            // Extract Q, K, V for this head
            const float* batch_qkv = qkv + b * seq_len * qkv_stride;
            
            // Initialize m and l (for numerical stability)
            float* m = m_buffer_.data() + head_idx * seq_len;
            float* l = l_buffer_.data() + head_idx * seq_len;
            float* o = output + b * seq_len * dim_ + h * head_dim_;
            
            std::fill(m, m + seq_len, -INFINITY);
            std::fill(l, l + seq_len, 0.0f);
            std::fill(o, o + seq_len * head_dim_, 0.0f);
            
            // Process in blocks for better cache usage
            const int BLOCK_SIZE = std::min(config_.block_size_q, seq_len);
            
            for (int q_start = 0; q_start < seq_len; q_start += BLOCK_SIZE) {
                int q_end = std::min(q_start + BLOCK_SIZE, seq_len);
                
                for (int k_start = 0; k_start < seq_len; k_start += BLOCK_SIZE) {
                    int k_end = std::min(k_start + BLOCK_SIZE, seq_len);
                    
                    // Compute attention for this block
                    for (int q_idx = q_start; q_idx < q_end; ++q_idx) {
                        const float* q = batch_qkv + q_idx * qkv_stride + h * head_dim_;
                        
                        float row_max = m[q_idx];
                        float row_sum = l[q_idx];
                        
                        // Compute scores for this query against all keys in block
                        std::vector<float> scores(k_end - k_start);
                        
                        for (int k_idx = k_start; k_idx < k_end; ++k_idx) {
                            const float* k = batch_qkv + k_idx * qkv_stride + dim_ + h * head_dim_;
                            
                            float score = 0.0f;
                            #pragma omp simd reduction(+:score)
                            for (int d = 0; d < head_dim_; ++d) {
                                score += q[d] * k[d];
                            }
                            score *= scale_;
                            
                            // Apply causal mask if needed
                            if (config_.causal_mask && k_idx > q_idx) {
                                score = -INFINITY;
                            }
                            
                            scores[k_idx - k_start] = score;
                            row_max = std::max(row_max, score);
                        }
                        
                        // Compute exponentials and sum
                        float block_sum = 0.0f;
                        for (int i = 0; i < k_end - k_start; ++i) {
                            scores[i] = std::exp(scores[i] - row_max);
                            block_sum += scores[i];
                        }
                        
                        // Update running statistics
                        float row_max_prev = m[q_idx];
                        float row_sum_new = row_sum * std::exp(row_max_prev - row_max) + block_sum;
                        
                        // Rescale previous output
                        if (row_sum_new > 0) {
                            float scale_prev = row_sum * std::exp(row_max_prev - row_max) / row_sum_new;
                            float scale_curr = 1.0f / row_sum_new;
                            
                            float* out_row = o + q_idx * dim_;
                            
                            // Scale previous output
                            #pragma omp simd
                            for (int d = 0; d < head_dim_; ++d) {
                                out_row[d] *= scale_prev;
                            }
                            
                            // Add contribution from current block
                            for (int k_idx = k_start; k_idx < k_end; ++k_idx) {
                                const float* v = batch_qkv + k_idx * qkv_stride + 2 * dim_ + h * head_dim_;
                                float attn_weight = scores[k_idx - k_start] * scale_curr;
                                
                                #pragma omp simd
                                for (int d = 0; d < head_dim_; ++d) {
                                    out_row[d] += attn_weight * v[d];
                                }
                            }
                        }
                        
                        m[q_idx] = row_max;
                        l[q_idx] = row_sum_new;
                    }
                }
            }
        }
    }
}

// ============================================================================
// MultiHeadFlashAttention Implementation
// ============================================================================

MultiHeadFlashAttention::MultiHeadFlashAttention(int dim, int num_heads,
                                                 const FlashAttentionConfig& config)
    : dim_(dim), num_heads_(num_heads) {
    
    qkv_proj_ = std::make_unique<LinearFlash>(dim, 3 * dim, false);
    attention_ = std::make_unique<FlashAttention>(dim, num_heads, config);
    out_proj_ = std::make_unique<LinearFlash>(dim, dim, false);
}

MultiHeadFlashAttention::~MultiHeadFlashAttention() = default;

void MultiHeadFlashAttention::forward(const float* input, float* output,
                                      int batch_size, int seq_len) const {
    // Ensure buffer is large enough
    size_t qkv_size = batch_size * seq_len * 3 * dim_;
    if (qkv_buffer_.size() < qkv_size) {
        qkv_buffer_.resize(qkv_size);
    }
    
    // Project to QKV
    qkv_proj_->forward(input, qkv_buffer_.data(), batch_size * seq_len);
    
    // Apply attention
    attention_->forward(qkv_buffer_.data(), output, batch_size, seq_len);
    
    // Output projection (in-place)
    out_proj_->forward(output, output, batch_size * seq_len);
}

int MultiHeadFlashAttention::num_parameters() const {
    return (3 * dim_ * dim_) + (dim_ * dim_);  // QKV + output projections
}

// ============================================================================
// SwiGLU Implementation
// ============================================================================

SwiGLU::SwiGLU(int dim, int hidden_dim)
    : dim_(dim), hidden_dim_(hidden_dim) {
    
    gate_proj_ = std::make_unique<LinearFlash>(dim, hidden_dim, false);
    up_proj_ = std::make_unique<LinearFlash>(dim, hidden_dim, false);
    down_proj_ = std::make_unique<LinearFlash>(hidden_dim, dim, false);
}

SwiGLU::~SwiGLU() = default;

void SwiGLU::forward(const float* input, float* output,
                    int batch_size, int seq_len) const {
    int total_tokens = batch_size * seq_len;
    
    // Ensure buffers are large enough
    size_t hidden_size = total_tokens * hidden_dim_;
    if (gate_buffer_.size() < hidden_size) {
        gate_buffer_.resize(hidden_size);
        up_buffer_.resize(hidden_size);
    }
    
    // Compute gate and up projections
    gate_proj_->forward(input, gate_buffer_.data(), total_tokens);
    up_proj_->forward(input, up_buffer_.data(), total_tokens);
    
    // Apply SwiGLU: gate * silu(up)
    #pragma omp parallel for
    for (int i = 0; i < total_tokens * hidden_dim_; ++i) {
        up_buffer_[i] = gate_buffer_[i] * silu(up_buffer_[i]);
    }
    
    // Down projection
    down_proj_->forward(up_buffer_.data(), output, total_tokens);
}

int SwiGLU::num_parameters() const {
    return 3 * dim_ * hidden_dim_;  // gate + up + down projections
}

// ============================================================================
// FlashTransformerBlock Implementation
// ============================================================================

FlashTransformerBlock::FlashTransformerBlock(int dim, int num_heads, int hidden_dim,
                                             const FlashAttentionConfig& config)
    : dim_(dim) {
    
    norm1_ = std::make_unique<RMSNorm>(dim);
    attention_ = std::make_unique<MultiHeadFlashAttention>(dim, num_heads, config);
    norm2_ = std::make_unique<RMSNorm>(dim);
    ffn_ = std::make_unique<SwiGLU>(dim, hidden_dim);
}

FlashTransformerBlock::~FlashTransformerBlock() = default;

void FlashTransformerBlock::forward(const float* input, float* output,
                                    int batch_size, int seq_len) const {
    int total_size = batch_size * seq_len * dim_;
    
    // Ensure buffers are large enough
    if (residual_.size() < total_size) {
        residual_.resize(total_size);
        norm_buffer_.resize(total_size);
        attn_output_.resize(total_size);
    }
    
    // Save residual
    std::copy(input, input + total_size, residual_.data());
    
    // norm1 -> attention -> add residual
    norm1_->forward(input, norm_buffer_.data(), batch_size * seq_len);
    attention_->forward(norm_buffer_.data(), attn_output_.data(), batch_size, seq_len);
    
    #pragma omp parallel for simd
    for (int i = 0; i < total_size; ++i) {
        output[i] = residual_[i] + attn_output_[i];
    }
    
    // Save new residual
    std::copy(output, output + total_size, residual_.data());
    
    // norm2 -> ffn -> add residual
    norm2_->forward(output, norm_buffer_.data(), batch_size * seq_len);
    ffn_->forward(norm_buffer_.data(), attn_output_.data(), batch_size, seq_len);
    
    #pragma omp parallel for simd
    for (int i = 0; i < total_size; ++i) {
        output[i] = residual_[i] + attn_output_[i];
    }
}

int FlashTransformerBlock::num_parameters() const {
    return 2 * dim_ +  // norms
           attention_->num_parameters() + 
           ffn_->num_parameters();
}

// ============================================================================
// SquareFlashTransformer Implementation
// ============================================================================

SquareFlashTransformer::SquareFlashTransformer(int num_layers, int dim, int num_heads,
                                               int hidden_dim, const FlashAttentionConfig& config)
    : dim_(dim), num_layers_(num_layers), 
      gradient_checkpointing_(false), inference_mode_(false) {
    
    layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        layers_.emplace_back(std::make_unique<FlashTransformerBlock>(
            dim, num_heads, hidden_dim, config
        ));
    }
    
    final_norm_ = std::make_unique<RMSNorm>(dim);
    
#ifdef USE_CUDA
    d_buffer1_ = nullptr;
    d_buffer2_ = nullptr;
#endif
}

SquareFlashTransformer::~SquareFlashTransformer() {
#ifdef USE_CUDA
    if (d_buffer1_) cudaFree(d_buffer1_);
    if (d_buffer2_) cudaFree(d_buffer2_);
#endif
}

void SquareFlashTransformer::forward(const float* input, float* output, int batch_size) const {
    const int seq_len = 64;  // Fixed for chess
    const int total_size = batch_size * seq_len * dim_;
    
    // Ensure buffers are large enough
    if (buffer1_.size() < total_size) {
        buffer1_.resize(total_size);
        buffer2_.resize(total_size);
    }
    
    // Copy input to first buffer
    std::copy(input, input + total_size, buffer1_.data());
    
    // Process through layers with ping-pong buffers
    float* src = buffer1_.data();
    float* dst = buffer2_.data();
    
    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(src, dst, batch_size, seq_len);
        std::swap(src, dst);
    }
    
    // Final norm
    final_norm_->forward(src, output, batch_size * seq_len);
}

int SquareFlashTransformer::num_parameters() const {
    int params = dim_;  // final norm
    for (const auto& layer : layers_) {
        params += layer->num_parameters();
    }
    return params;
}

// ============================================================================
// RotaryEmbedding Implementation
// ============================================================================

RotaryEmbedding::RotaryEmbedding(int dim, int max_seq_len, float base)
    : dim_(dim), max_seq_len_(max_seq_len), base_(base) {
    
    compute_cache();
    
#ifdef USE_CUDA
    d_cos_cached_ = nullptr;
    d_sin_cached_ = nullptr;
#endif
}

RotaryEmbedding::~RotaryEmbedding() {
#ifdef USE_CUDA
    if (d_cos_cached_) cudaFree(d_cos_cached_);
    if (d_sin_cached_) cudaFree(d_sin_cached_);
#endif
}

void RotaryEmbedding::compute_cache() {
    cos_cached_.resize(max_seq_len_ * dim_ / 2);
    sin_cached_.resize(max_seq_len_ * dim_ / 2);
    
    for (int seq_idx = 0; seq_idx < max_seq_len_; ++seq_idx) {
        for (int i = 0; i < dim_ / 2; ++i) {
            float freq = 1.0f / std::pow(base_, 2.0f * i / dim_);
            float angle = seq_idx * freq;
            cos_cached_[seq_idx * dim_ / 2 + i] = std::cos(angle);
            sin_cached_[seq_idx * dim_ / 2 + i] = std::sin(angle);
        }
    }
}

void RotaryEmbedding::apply(float* q, float* k, int batch_size, int seq_len, int num_heads) const {
    const int head_dim = dim_ / num_heads;
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                float* q_head = q + b * seq_len * dim_ + s * dim_ + h * head_dim;
                float* k_head = k + b * seq_len * dim_ + s * dim_ + h * head_dim;
                
                const float* cos = cos_cached_.data() + s * head_dim / 2;
                const float* sin = sin_cached_.data() + s * head_dim / 2;
                
                // Apply rotation
                for (int i = 0; i < head_dim / 2; ++i) {
                    float q1 = q_head[i];
                    float q2 = q_head[i + head_dim / 2];
                    float k1 = k_head[i];
                    float k2 = k_head[i + head_dim / 2];
                    
                    q_head[i] = q1 * cos[i] - q2 * sin[i];
                    q_head[i + head_dim / 2] = q1 * sin[i] + q2 * cos[i];
                    k_head[i] = k1 * cos[i] - k2 * sin[i];
                    k_head[i + head_dim / 2] = k1 * sin[i] + k2 * cos[i];
                }
            }
        }
    }
}

} // namespace stac::model
