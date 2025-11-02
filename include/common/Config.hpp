#pragma once

#include <string>
#include <filesystem>

namespace stac {

// ============================================================================
// Model Configuration
// ============================================================================

struct ModelConfig {
    // Transformer architecture
    int embedding_dim = 256;        // Token dimension (d)
    int num_layers = 8;              // Number of transformer blocks (L)
    int num_heads = 8;               // Number of attention heads
    int mlp_hidden_dim = 1024;       // Hidden dimension in MLP
    
    // Regularization
    float dropout = 0.0f;            // Dropout rate (if training)
    float layernorm_eps = 1e-5f;     // LayerNorm epsilon
    
    // Activation
    bool use_gelu = true;            // GELU vs ReLU in MLP
    
    // Device
    bool use_cuda = true;            // Enable CUDA acceleration
    int cuda_device = 0;             // GPU device ID
    
    // Validate configuration
    void validate() const {
        if (embedding_dim % num_heads != 0) {
            throw std::invalid_argument("embedding_dim must be divisible by num_heads");
        }
        if (num_layers < 1) {
            throw std::invalid_argument("num_layers must be >= 1");
        }
    }
};

// ============================================================================
// Training Configuration
// ============================================================================

struct TrainingConfig {
    // PPO hyperparameters
    float learning_rate = 3e-4f;
    float clip_epsilon = 0.2f;       // PPO clip parameter
    float value_loss_coef = 0.5f;    // Value loss coefficient
    float entropy_coef = 0.01f;      // Entropy bonus coefficient
    
    // GAE parameters
    float gamma = 0.99f;             // Discount factor
    float gae_lambda = 0.95f;        // GAE lambda
    
    // Training schedule
    int num_epochs = 4;              // PPO epochs per update
    int batch_size = 4096;           // Minibatch size
    int rollout_length = 2048;       // Steps per rollout
    int num_envs = 128;              // Parallel environments
    
    // Optimizer
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;
    float grad_clip_norm = 0.5f;    // Gradient clipping
    
    // Learning rate schedule
    bool use_lr_schedule = true;
    float lr_schedule_gamma = 0.99f; // Exponential decay
    int lr_schedule_step = 1000;     // Steps between decay
    
    // Checkpointing
    int checkpoint_interval = 10000; // Steps between checkpoints
    std::filesystem::path checkpoint_dir = "./checkpoints";
    
    // Logging
    int log_interval = 10;           // Steps between log messages
    
    // Validation
    int validation_games = 100;      // Games for validation
    int validation_interval = 5000;  // Steps between validation
};

// ============================================================================
// Environment Configuration
// ============================================================================

struct EnvironmentConfig {
    // Adapter settings
    std::string adapter_type = "lichess";  // "lichess", "uci", "stockfish"
    std::string server_url = "ws://localhost:8080";
    int connection_timeout_ms = 5000;
    
    // Game settings
    int max_moves = 512;             // Maximum moves per game
    float time_per_move_ms = 100.0f; // Time allocation
    bool use_opening_book = false;   // Use opening book
    int opening_book_depth = 10;     // Depth in opening book
    
    // Exploration
    float dirichlet_alpha = 0.3f;    // Dirichlet noise for root
    float dirichlet_epsilon = 0.25f; // Mixing parameter
    float temperature = 1.0f;        // Action sampling temperature
    
    // Self-play
    bool add_noise_to_policy = true; // Add exploration noise
    int num_simulations = 800;       // MCTS simulations (if using)
};

// ============================================================================
// Inference Configuration
// ============================================================================

struct InferenceConfig {
    // Batch processing
    int max_batch_size = 256;        // Maximum batch size
    int batch_timeout_ms = 10;       // Timeout for batch accumulation
    
    // Caching
    bool use_cache = true;           // Cache recent positions
    int cache_size = 10000;          // Number of positions to cache
    
    // Temperature schedule
    float temperature_start = 1.0f;
    float temperature_end = 0.1f;
    int temperature_moves = 30;      // Moves to decay temperature
    
    // Output
    bool return_policy_probs = true; // Return full policy distribution
    bool return_top_k_moves = 5;     // Number of top moves to return
};

// ============================================================================
// Master Configuration
// ============================================================================

struct Config {
    ModelConfig model;
    TrainingConfig training;
    EnvironmentConfig environment;
    InferenceConfig inference;
    
    // Logging
    std::string log_level = "INFO";  // DEBUG, INFO, WARNING, ERROR
    std::filesystem::path log_dir = "./logs";
    bool use_tensorboard = true;
    int log_interval = 100;          // Steps between logs
    
    // Load from file
    static Config from_yaml(const std::filesystem::path& path);
    static Config from_json(const std::filesystem::path& path);
    
    // Save to file
    void to_yaml(const std::filesystem::path& path) const;
    void to_json(const std::filesystem::path& path) const;
    
    // Validate all configurations
    void validate() const {
        model.validate();
        // Add more validation as needed
    }
};

} // namespace stac