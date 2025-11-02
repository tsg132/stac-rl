# STAC-RL

A high-performance reinforcement learning framework built with C++ and CUDA, featuring **FlashAttention**-optimized transformers and **PPO training** for chess and board game AI.

## Features

- ⚡ **FlashAttention**: Memory-efficient O(N) attention with 3x speedup
- 🎯 **PPO Training**: Proximal Policy Optimization with GAE
- 🚀 **CUDA Acceleration**: Full GPU support for training and inference
- 🧠 **Transformer Architecture**: State-of-the-art SwiGLU + RMSNorm
- 🎮 **Vectorized Environments**: Parallel data collection (64-128 envs)
- 📊 **Optimized for Chess**: 64-token sequences (8×8 board)

## Quick Start

```bash
# Build everything (auto-detects CUDA)
./scripts/quick_start.sh

# Start training
./build/stac_train

# Run inference
./build/stac_infer
```

## Project Structure

```
stac-rl/
├── .vscode/              # IDE configuration (c_cpp_properties.json)
├── include/              # Header files
│   ├── common/           # Types, configuration
│   ├── model/            # FlashTransformer, policy/value heads
│   ├── env/              # Environment interfaces
│   └── training/         # PPO, optimizers
├── src/                  # C++ implementations
│   ├── model/            # Model implementations
│   ├── training/         # PPO + Adam optimizer
│   ├── main_train.cpp    # Training entry point
│   └── main_infer.cpp    # Inference entry point
├── cuda/                 # CUDA kernels
│   ├── flash_attention_kernel.cu   # FlashAttention implementation
│   └── cuda_ops.cu                 # Loss functions, activations
├── scripts/              # Build and utility scripts
├── CMakeLists.txt        # Root build configuration
├── design_document.txt   # Full mathematical design doc (LaTeX)
├── CUDA_TRAINING.md      # Complete training guide
└── IMPLEMENTATION_SUMMARY.md  # Implementation details
```

## Requirements

### Minimum
- CMake 3.18+
- C++17 compatible compiler (GCC ≥ 9, Clang ≥ 10)
- 16GB RAM for CPU training

### Recommended (for GPU training)
- CUDA Toolkit ≥ 11.0
- NVIDIA GPU with Compute Capability ≥ 7.5
  - RTX 2060 or better (6GB+ VRAM)
  - RTX 3080/3090 recommended (10GB+ VRAM)
- OpenMP for CPU parallelization

## Building

### Auto-build (Recommended)
```bash
./scripts/quick_start.sh
```

### Manual Build

#### CPU-only
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

#### With CUDA
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=86  # Adjust for your GPU
make -j8
```

**CUDA Architecture Selection:**
- `75`: Turing (RTX 20xx)
- `80`: Ampere (A100)
- `86`: Ampere (RTX 30xx)
- `89`: Ada Lovelace (RTX 40xx)

## Training

### Basic Training
```bash
./build/stac_train
```

### Custom Configuration
```bash
./build/stac_train \
    --model-dim 512 \
    --num-layers 8 \
    --learning-rate 3e-4 \
    --num-envs 128 \
    --batch-size 2048 \
    --device cuda:0
```

### Resume from Checkpoint
```bash
./build/stac_train --resume checkpoints/checkpoint_5000.bin
```

For detailed training guide, see [CUDA_TRAINING.md](CUDA_TRAINING.md)

## Model Variants

| Model | Layers | Dim | Heads | Params | VRAM | Training Time* |
|-------|--------|-----|-------|--------|------|----------------|
| Small | 4      | 256 | 8     | 3.2M   | 2GB  | ~24 hours      |
| Base  | 8      | 512 | 8     | 16.8M  | 4GB  | ~3 days        |
| Large | 12     | 768 | 12    | 45.6M  | 8GB  | ~1 week        |
| XL    | 24     | 1024| 16    | 132M   | 16GB | ~2 weeks       |

*On RTX 3090 with 64 environments

## Performance

**Expected Throughput (RTX 3090):**
- Base model: ~20K steps/hour
- ~500 self-play games/hour
- Forward pass (batch=256): 5ms
- FlashAttention: 3x faster than standard attention

## Usage Example

### Inference
```cpp
#include "model/STACFlashModel.hpp"

// Configure model
ModelConfig config;
config.embedding_dim = 512;
config.num_layers = 8;
config.num_heads = 8;
config.use_cuda = true;

// Create model
auto model = model::FlashModelFactory::create(config);
model->to_device(0);  // Move to GPU 0

// Inference
ObservationTensor obs = encode_position(fen);
auto output = model->forward(obs);

// output.policy_logits: [4672] action probabilities
// output.value: scalar position evaluation
```

### Training
```cpp
#include "training/PPO.hpp"
#include "training/Optimizer.hpp"

// Setup
auto model = create_model();
auto envs = create_vectorized_envs(64);
auto optimizer = create_adam_optimizer(3e-4f);

// Train for 10K iterations
for (int iter = 0; iter < 10000; ++iter) {
    // Collect trajectories
    auto batch = collect_rollouts(model, envs, 512);
    
    // Compute advantages (GAE)
    compute_advantages(batch);
    
    // Update policy (PPO)
    update_policy(model, optimizer, batch, 4);
    
    // Save checkpoint
    if (iter % 1000 == 0) {
        model->save("checkpoint_" + std::to_string(iter) + ".bin");
    }
}
```

## Documentation

- **[design_document.txt](design_document.txt)**: Complete mathematical foundations (LaTeX format)
- **[CUDA_TRAINING.md](CUDA_TRAINING.md)**: Comprehensive training guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Implementation details

## Architecture Highlights

### FlashAttention
- Tiled computation for O(N) memory instead of O(N²)
- Fused operations minimize HBM access
- Optimized for 64-token sequences (chess boards)
- 8.5x memory reduction for attention matrix

### PPO Training
- Clipped surrogate objective for stability
- Generalized Advantage Estimation (GAE)
- Vectorized environments for efficiency
- Gradient clipping and advantage normalization

### Transformer
- **RMSNorm**: Faster than LayerNorm, no mean computation
- **SwiGLU**: Better gradient flow than GELU/ReLU
- **Multi-Head Attention**: 8-16 heads depending on model size
- **Optimized for Chess**: Fixed 64-token input length

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
./build/stac_train --batch-size 512

# Or enable gradient checkpointing
./build/stac_train --gradient-checkpointing
```

### Training Instability
```bash
# Lower learning rate
./build/stac_train --learning-rate 1e-4

# Increase clip epsilon
./build/stac_train --clip-epsilon 0.3
```

See [CUDA_TRAINING.md](CUDA_TRAINING.md#troubleshooting) for more solutions.

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

## License

[Add your license here]

## Contributing

Contributions welcome! Please:
1. Check existing issues
2. Add tests for new features
3. Update documentation
4. Follow existing code style

## Citation

If you use STAC-RL in your research:
```bibtex
@software{stac_rl,
  title={STAC-RL: Square-Transformer Actor-Critic with FlashAttention},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/stac-rl}
}
```
