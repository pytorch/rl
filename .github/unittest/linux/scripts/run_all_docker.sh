#!/usr/bin/env bash
# Simplified test script for use with custom Docker images
# Assumes PyTorch, tensordict, and test dependencies are pre-installed

set -euxo pipefail
set -v

echo "============================================"
echo "Running TorchRL tests with pre-built image"
echo "============================================"

# Activate pre-installed virtual environment
if [ -d "/opt/venv" ]; then
    source /opt/venv/bin/activate
    echo "✓ Activated pre-installed environment at /opt/venv"
else
    echo "✗ ERROR: Pre-installed environment not found at /opt/venv"
    echo "This script requires a custom Docker image with pre-installed dependencies"
    exit 1
fi

# Git setup
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
cd "${root_dir}"

# Environment variables
export PYTORCH_TEST_WITH_SLOW='1'
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=100
export BATCHED_PIPE_TIMEOUT=60
export LAZY_LEGACY_OP=False
export RL_LOGGING_LEVEL=DEBUG
export TOKENIZERS_PARALLELISM=true

# Set rendering backend
if [ "${CU_VERSION:-}" == cpu ] ; then
  export MUJOCO_GL=glfw
else
  export MUJOCO_GL=egl
fi
export SDL_VIDEODRIVER=dummy
export PYOPENGL_PLATFORM=$MUJOCO_GL
export DISPLAY=:99

# Verify pre-installed dependencies
echo "Verifying pre-installed dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensordict; print(f'tensordict: {tensordict.__version__}')"
python -c "import gymnasium; print('gymnasium: OK')"
python -c "import dm_control; print('dm_control: OK')"

# Update git submodules (lightweight operation)
echo "Updating git submodules..."
git submodule sync && git submodule update --init --recursive

# Install TorchRL from source (this is the main build step)
echo "Installing TorchRL from source..."
uv pip install -e . --no-build-isolation --no-deps

# Verify TorchRL installation
python -c "import torchrl; print(f'TorchRL: {torchrl.__version__}')"

# Install VC1 models for GPU tests (downloads pre-trained models)
if [ "${CU_VERSION:-}" != cpu ] ; then
  echo "Installing VC1 models..."
  python -c """
from torchrl.envs.transforms.vc1 import VC1Transform
VC1Transform.install_vc_models(auto_exit=True)
"""
  python -c "import vc_models; print('VC1 models: OK')"
fi

# Start Xvfb for headless rendering
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x24 &
sleep 2

# Display environment info
echo "============================================"
echo "Environment Information"
echo "============================================"
python -m torch.utils.collect_env

# Run smoke tests first
echo "============================================"
echo "Running smoke tests"
echo "============================================"
pytest test/smoke_test.py -v --durations 200
pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'

# Run full test suite
echo "============================================"
echo "Running full test suite"
echo "============================================"
if [ "${CU_VERSION:-}" != cpu ] ; then
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no \
    --ignore test/test_rlhf.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda
else
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no \
    --ignore test/test_rlhf.py \
    --ignore test/test_distributed.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda
fi

# Combine coverage results
coverage combine -q
coverage xml -i

echo "============================================"
echo "Tests completed successfully!"
echo "============================================"

