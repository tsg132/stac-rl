# Quick Start on Colab

## 1. Enable GPU
**Runtime → Change runtime type → GPU → Save**

## 2. Run this in one cell:

```python
# Clone and setup
!git clone https://github.com/tsg132/stac-rl.git
%cd stac-rl

# Install dependencies
!apt-get update -qq && apt-get install -y cmake build-essential
!pip install -q python-chess numpy torch

# Build with CUDA
!mkdir -p build
%cd build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!make -j$(nproc)
%cd ..
```

## 3. If build fails, check:

```python
# Verify GPU
!nvidia-smi

# Check CUDA compiler
!nvcc --version

# Try clean build
!rm -rf build && mkdir build
%cd build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!make -j$(nproc) VERBOSE=1  # See full error messages
```

## 4. Train (PyTorch - most reliable):

```python
# Generate data
!mkdir -p data
!python3 scripts/self_play.py --num-games 100 --output data/games.npz

# Train
!python3 scripts/train_pytorch.py \
    --data data/games.npz \
    --device cuda \
    --epochs 10 \
    --batch-size 256 \
    --d-model 512 \
    --n-layers 8
```

## Common Issues

**"fatal error: cuda_runtime.h: No such file or directory"**
- Make sure GPU runtime is enabled
- Run `!nvidia-smi` to verify
- Try rebuilding from scratch

**"CUDA out of memory"**
- Reduce batch size: `--batch-size 128`
- Reduce model size: `--d-model 256 --n-layers 4`

**Session disconnects**
- Colab free tier has ~12 hour limit
- Save checkpoints frequently
