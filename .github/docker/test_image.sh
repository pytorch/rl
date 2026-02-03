#!/bin/bash
# Test script to verify Docker image functionality

set -e

IMAGE=${1:-"ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest"}

echo "Testing Docker image: $IMAGE"
echo "================================"

# Test 1: Image exists and can be pulled
echo "Test 1: Pulling image..."
docker pull $IMAGE

# Test 2: Python environment
echo ""
echo "Test 2: Python environment..."
docker run --rm $IMAGE python --version

# Test 3: PyTorch
echo ""
echo "Test 3: PyTorch installation..."
docker run --rm $IMAGE python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test 4: tensordict
echo ""
echo "Test 4: tensordict installation..."
docker run --rm $IMAGE python -c "import tensordict; print(f'tensordict: {tensordict.__version__}')"

# Test 5: Key dependencies
echo ""
echo "Test 5: Key dependencies..."
docker run --rm $IMAGE python -c "
import gymnasium
import dm_control
import pytest
import numpy
import transformers
print('✓ All key dependencies available')
"

# Test 6: Build tools
echo ""
echo "Test 6: Build tools..."
docker run --rm $IMAGE bash -c "
which uv && echo '✓ uv available' && \
which ninja && echo '✓ ninja available' && \
which cmake && echo '✓ cmake available'
"

# Test 7: System libraries
echo ""
echo "Test 7: System libraries..."
docker run --rm $IMAGE bash -c "
ldconfig -p | grep libGL && echo '✓ OpenGL libraries available' && \
ldconfig -p | grep libegl && echo '✓ EGL libraries available'
"

# Test 8: Environment variables
echo ""
echo "Test 8: Environment variables..."
docker run --rm $IMAGE bash -c "
echo 'MUJOCO_GL='$MUJOCO_GL && \
echo 'DISPLAY='$DISPLAY && \
echo 'PATH='$PATH
"

# Test 9: Virtual environment
echo ""
echo "Test 9: Virtual environment..."
docker run --rm $IMAGE bash -c "
source /opt/venv/bin/activate && \
python -c 'import sys; print(f\"Python: {sys.executable}\")' && \
pip list | head -20
"

echo ""
echo "================================"
echo "✅ All tests passed for $IMAGE"
echo "================================"

