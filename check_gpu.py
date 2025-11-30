"""
GPU Verification Script
Checks if GPU is available and properly configured for YOLO training.
"""

import torch
import sys

print("=" * 70)
print(" GPU Verification Check")
print("=" * 70)

# Check PyTorch CUDA availability
print("\n1. PyTorch CUDA Check:")
print(f"   PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"     Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"     Compute Capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print("\n2. GPU Computation Test:")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU computation test: PASSED")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ GPU computation test: FAILED")
        print(f"      Error: {e}")
    
    # Check if YOLO can use GPU
    print("\n3. YOLO GPU Check:")
    try:
        from ultralytics import YOLO
        # Try to create a dummy model and check device
        print("   ✅ Ultralytics YOLO imported successfully")
        print("   Note: YOLO will automatically use GPU if available")
    except Exception as e:
        print(f"   ⚠️ Ultralytics import issue: {e}")
else:
    print("\n   ❌ CUDA is NOT available!")
    print("\n   Possible reasons:")
    print("   1. PyTorch was installed without CUDA support")
    print("   2. CUDA drivers are not installed")
    print("   3. GPU is not compatible")
    print("\n   Solutions:")
    print("   1. Install CUDA-enabled PyTorch:")
    print("      Visit: https://pytorch.org/get-started/locally/")
    print("   2. Check NVIDIA drivers:")
    print("      Run: nvidia-smi")
    print("   3. Verify CUDA installation:")
    print("      Run: nvcc --version")

# Check nvidia-smi
print("\n4. NVIDIA Driver Check:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✅ nvidia-smi is available")
        # Extract GPU info from nvidia-smi
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA-SMI' in line or 'Driver Version' in line:
                print(f"   {line.strip()}")
    else:
        print("   ⚠️ nvidia-smi returned error")
except FileNotFoundError:
    print("   ❌ nvidia-smi not found - NVIDIA drivers may not be installed")
except Exception as e:
    print(f"   ⚠️ Could not run nvidia-smi: {e}")

print("\n" + "=" * 70)
print(" Summary:")
if torch.cuda.is_available():
    print(" ✅ GPU is available and ready for training")
    print("    Training should use GPU automatically")
else:
    print(" ❌ GPU is NOT available")
    print("    Training will run on CPU (VERY SLOW - 5-10+ hours)")
    print("    Install CUDA-enabled PyTorch to use GPU")
print("=" * 70)

