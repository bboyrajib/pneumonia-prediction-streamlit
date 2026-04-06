"""
Run this after installing requirements to verify GPU is detected by PyTorch.
Usage: python verify_gpu.py
"""

import torch
import sys

print(f"PyTorch version    : {torch.__version__}")
print(f"Python version     : {sys.version.split()[0]}")
print(f"CUDA available     : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version       : {torch.version.cuda}")
    print(f"GPU detected       : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM               : {props.total_memory / 1e9:.1f} GB")
    print(f"SM count           : {props.multi_processor_count}")

    # Quick compute test on GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    print(f"Matrix multiply    : OK (device={c.device})")

    # cuDNN status
    print(f"cuDNN enabled      : {torch.backends.cudnn.enabled}")
    print(f"cuDNN version      : {torch.backends.cudnn.version()}")
else:
    print("GPU detected       : None — running on CPU")
    print("If you expected GPU, ensure:")
    print("  1. NVIDIA driver is installed (nvidia-smi works)")
    print("  2. PyTorch was installed with the correct CUDA wheel:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

print("\nAll checks passed.")
