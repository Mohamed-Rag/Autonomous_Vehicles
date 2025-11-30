# Training Speed Optimization Guide

## Current Issue: Training on CPU (Very Slow)

If you see `GPU_mem 0G` in training output, your model is running on CPU, which is **10-50x slower** than GPU.

---

## Quick Fixes

### 1. Verify GPU is Available

Run the GPU check script:
```bash
python check_gpu.py
```

This will tell you:
- ✅ If GPU is detected
- ✅ If CUDA is properly installed
- ✅ If PyTorch can use GPU

### 2. Install CUDA-Enabled PyTorch

If GPU check shows CUDA is not available, install CUDA-enabled PyTorch:

**For Windows:**
```bash
# Check your CUDA version first (run: nvidia-smi)
# Then install matching PyTorch from: https://pytorch.org/get-started/locally/

# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify installation:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
```

### 3. Restart Training

After fixing GPU issues, restart training:
```bash
python train.py
```

You should now see:
- `GPU_mem X.XG` (not 0G)
- Much faster training (minutes instead of hours)

---

## Speed Optimizations Applied

The updated `train.py` now includes:

### ✅ Automatic GPU Detection
- Verifies GPU is working before training
- Falls back to CPU with warning if GPU unavailable

### ✅ Optimized Workers
- Windows: Uses 0 workers (multiprocessing issues)
- Linux: Uses 4-8 workers for faster data loading

### ✅ Smart Caching
- Only caches images if enough RAM available
- Prevents out-of-memory errors

### ✅ Batch Size Optimization
- Automatically adjusts based on VRAM
- Larger batches = faster training

### ✅ Mixed Precision (AMP)
- Enabled on GPU for 2x speedup
- Disabled on CPU (not supported)

---

## Expected Training Times

### With GPU (NVIDIA):
- **RTX 3060 (8GB)**: ~30-60 minutes for 1 epoch
- **RTX 3080 (10GB)**: ~20-40 minutes for 1 epoch
- **RTX 4090 (24GB)**: ~10-20 minutes for 1 epoch

### Without GPU (CPU):
- **Modern CPU (8+ cores)**: 5-10+ hours for 1 epoch
- **Older CPU**: 10-20+ hours for 1 epoch

---

## Troubleshooting

### Issue: GPU shows 0G but nvidia-smi works

**Solution:**
1. Reinstall PyTorch with CUDA support
2. Restart Python/terminal
3. Verify: `python check_gpu.py`

### Issue: Out of Memory (OOM) errors

**Solution:**
1. Reduce batch size in `train.py`
2. Reduce image size (imgsz=640 instead of 800)
3. Disable cache: `cache=False`

### Issue: Workers = 0 on Windows

**This is normal!** Windows has multiprocessing issues with YOLO. The code automatically sets workers=0 on Windows to prevent crashes.

### Issue: Training still slow with GPU

**Check:**
1. GPU utilization: Run `nvidia-smi` during training
2. Batch size: Increase if VRAM allows
3. Image size: Reduce if needed (640 is faster than 800)
4. Other processes: Close other GPU-using applications

---

## Manual Speed Tweaks

If you want to manually adjust settings in `train.py`:

```python
# Increase batch size (if you have VRAM)
batch_size = 16  # or 32 if you have 24GB+ VRAM

# Reduce image size (faster training)
imgsz = 640  # instead of 800

# Reduce freeze layers (faster but may reduce accuracy)
freeze_layers = 2  # instead of 4

# Disable validation during training (faster, but less monitoring)
val = False  # Only for speed testing
```

---

## Monitoring Training Speed

Watch these metrics:
- **it/s**: Iterations per second (higher = faster)
- **GPU_mem**: GPU memory usage (should be >0)
- **ETA**: Estimated time remaining

If `it/s` is very low (<0.5), you're likely on CPU.

---

## Next Steps

1. ✅ Run `python check_gpu.py` to verify GPU
2. ✅ Install CUDA PyTorch if needed
3. ✅ Restart training with `python train.py`
4. ✅ Verify GPU is being used (check GPU_mem > 0)
5. ✅ Monitor training speed (should be much faster)

Good luck! 🚀

