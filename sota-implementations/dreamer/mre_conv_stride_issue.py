"""
Minimal Reproducible Example: torch.compile inductor convolution_backward stride mismatch

BUG: When using torch.compile(backend='inductor') on MPS or CUDA with a chain of
ConvTranspose2d layers, the backward pass fails with a stride assertion error.

The inductor's fake/meta kernel for aten.convolution_backward.default predicts
NHWC-like strides, but the actual CUDA/MPS ops produce NCHW contiguous tensors.

ERROR:
    AssertionError: expected size 64==64, stride 1==169 at dim=1; ...
    Error in op: torch.ops.aten.convolution_backward.default

AFFECTED: MPS, CUDA
WORKS ON: CPU

ENVIRONMENT:
    - PyTorch 2.9.1
    - Tested on MPS (Apple Silicon) and CUDA

To run:
    python mre_conv_stride_issue.py
"""

import torch
import torch.nn as nn


class ConvTransposeChain(nn.Module):
    """Chain of ConvTranspose2d layers (like Dreamer decoder)."""
    
    def __init__(self):
        super().__init__()
        # 4-layer decoder: 1024 -> 128 -> 64 -> 32 -> 3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
        )
    
    def forward(self, x):
        return self.decoder(x)


def test_device(device: torch.device) -> bool:
    """Test if inductor backward works on the given device."""
    model = ConvTransposeChain().to(device)
    compiled = torch.compile(model, backend='inductor')
    
    # Input: (batch, 1024, 1, 1)
    x = torch.randn(100, 1024, 1, 1, device=device)
    
    try:
        out = compiled(x)
        out.sum().backward()
        return True
    except AssertionError as e:
        if "convolution_backward" in str(e):
            return False
        raise


def main():
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    results = {}
    
    # Test CPU
    print("Testing CPU...")
    results["cpu"] = test_device(torch.device("cpu"))
    print(f"  CPU: {'PASS ✅' if results['cpu'] else 'FAIL ❌'}")
    
    # Test MPS
    if torch.backends.mps.is_available():
        print("Testing MPS...")
        results["mps"] = test_device(torch.device("mps"))
        print(f"  MPS: {'PASS ✅' if results['mps'] else 'FAIL ❌'}")
    
    # Test CUDA
    if torch.cuda.is_available():
        print("Testing CUDA...")
        results["cuda"] = test_device(torch.device("cuda"))
        print(f"  CUDA: {'PASS ✅' if results['cuda'] else 'FAIL ❌'}")
    
    # Summary
    print()
    if results.get("cpu") and not (results.get("mps", True) and results.get("cuda", True)):
        print("="*65)
        print("BUG CONFIRMED: inductor convolution_backward stride mismatch")
        print()
        print("The inductor's fake kernel for aten.convolution_backward.default")
        print("predicts incorrect (NHWC-like) output strides on MPS/CUDA.")
        print("The actual ops produce NCHW contiguous tensors.")
        print()
        print("Works on CPU, fails on MPS/CUDA.")
        print("="*65)


if __name__ == "__main__":
    main()
