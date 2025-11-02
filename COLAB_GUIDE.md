# Running STAC-RL on Google Colab with GPU

## Quick Start (3 steps)

### 1. Open Colab & Enable GPU
- Go to https://colab.research.google.com/
- Upload `colab_setup.ipynb` OR create a new notebook
- **Runtime → Change runtime type → GPU (T4 recommended)**

### 2. Clone & Build
```python
# In first Colab cell:
!git clone https://github.com/YOUR_USERNAME/stac-rl.git
%cd stac-rl

# Check GPU
!nvidia-smi

# Install dependencies
!apt-get update -qq && apt-get install -y cmake
!pip install python-chess numpy torch

# Build with CUDA
!mkdir -p build && cd build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!make -j$(nproc)
```

### 3. Train
```python
# Generate data (optional, or use existing)
!python3 scripts/self_play.py --num-games 1000 --output data/games.npz

# Train with PyTorch on GPU
!python3 scripts/train_pytorch.py \
    --data data/games.npz \
    --device cuda \
    --epochs 20 \
    --batch-size 256
```

## What GPU Will You Get?

Colab provides these GPUs (free tier):
- **T4** (16GB) - Most common, CUDA compute 7.5
- **P100** (16GB) - Older but powerful, CUDA compute 6.0
- **K80** (12GB) - Oldest, CUDA compute 3.7

Premium Colab ($10/month):
- **V100** (16GB) - CUDA compute 7.0
- **A100** (40GB) - Best available, CUDA compute 8.0

## Expected Training Times (T4 GPU)

- **1000 games**: ~2-3 minutes
- **10,000 games**: ~20-30 minutes
- **100,000 games**: ~3-4 hours

Model training (10K positions):
- **10 epochs**: ~5-10 minutes
- **50 epochs**: ~25-50 minutes

## CUDA Architecture Numbers

When compiling, use the right architecture for your GPU:
```bash
# Check your GPU
!nvidia-smi --query-gpu=compute_cap --format=csv

# Then use in cmake:
# K80:  -DCMAKE_CUDA_ARCHITECTURES=37
# P100: -DCMAKE_CUDA_ARCHITECTURES=60
# V100: -DCMAKE_CUDA_ARCHITECTURES=70
# T4:   -DCMAKE_CUDA_ARCHITECTURES=75
# A100: -DCMAKE_CUDA_ARCHITECTURES=80
```

## Memory Management

If you get OOM (Out of Memory):
```python
# Reduce batch size
--batch-size 128  # instead of 256

# Reduce model size
--d-model 256     # instead of 512
--n-layers 4      # instead of 8
```

## Saving Your Work

Download trained model:
```python
from google.colab import files
files.download('checkpoints/model_final.pt')
```

Or save to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp checkpoints/model_final.pt /content/drive/MyDrive/
```

## Tips

1. **Keep session alive**: Colab disconnects after ~30 mins idle
2. **Monitor GPU**: Run `!nvidia-smi` to check utilization
3. **Use small test first**: Try 100 games before running 100K
4. **Save frequently**: Download checkpoints regularly
5. **Check runtime**: Free tier has ~12 hour limit per session

## Troubleshooting

**"No CUDA GPUs available"**
→ Runtime → Change runtime type → GPU

**"CUDA out of memory"**
→ Reduce batch size or model dimensions

**"Session crashed"**
→ You used too much memory, reduce batch size

**"Compilation failed"**
→ Check CUDA architecture matches your GPU
→ Run `!nvcc --version` to verify CUDA toolkit

## Next Steps After Training

1. Download your trained model
2. Test inference locally or in Colab
3. Generate more games with trained model
4. Iterate: train → evaluate → improve
