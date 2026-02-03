#!/bin/bash
set -e

# Parse command line arguments
NO_BUILD=false
USE_NIGHTLY=false
BUILD_ONLY=false
RUN_SGLANG_TESTS=false
for arg in "$@"; do
    case $arg in
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --nightly)
            USE_NIGHTLY=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --sglang-tests)
            RUN_SGLANG_TESTS=true
            shift
            ;;
    esac
done

if [ "$NO_BUILD" = true ]; then
    echo "Running in --no-build mode: will pull repos but skip building"
fi
if [ "$USE_NIGHTLY" = true ]; then
    echo "Running in --nightly mode: will use PyTorch nightly wheels instead of building from source"
fi
if [ "$BUILD_ONLY" = true ]; then
    echo "Running in --build-only mode: will build but skip running dreamer"
fi
if [ "$RUN_SGLANG_TESTS" = true ]; then
    echo "Running in --sglang-tests mode: will run SGLang tests"
fi

cd /root/

# Configure git credentials for GitHub (set GH_TOKEN env var before running)
if [ -n "$GH_TOKEN" ]; then
    git config --global credential.helper store
    echo "https://vmoens:${GH_TOKEN}@github.com" > ~/.git-credentials
    echo "Git credentials configured for vmoens"
fi

# Setup ccache for faster rebuilds
export USE_CCACHE=1
export CCACHE_DIR=/root/.ccache
export PATH="/usr/lib/ccache:$PATH"
mkdir -p $CCACHE_DIR

# Install ccache if not present
if ! command -v ccache &> /dev/null; then
    apt-get update && apt-get install -y ccache cmake
fi

# Create venv only if it doesn't exist
if [ ! -d "/root/torchrl" ]; then
    uv venv torchrl --python 3.12
fi
source torchrl/bin/activate

# Helper function to safely pull a branch, with fallback to delete/refetch/checkout
safe_git_pull() {
    local branch=$1
    local remote_branch=$2
    if ! git pull; then
        echo "Git pull failed, resetting branch..."
        git checkout HEAD~0  # detach HEAD
        git branch -D "$branch" || true
        git fetch origin
        git checkout -b "$branch" "origin/$remote_branch"
    fi
}

# ============================================================
# PyTorch Installation: Nightly wheels OR build from source
# ============================================================
uv pip install "pybind11[global]"
if [ "$USE_NIGHTLY" = true ]; then
    # Use pre-built nightly wheels (includes cuDNN support)
    echo "=========================================="
    echo "Installing PyTorch from nightly wheels..."
    echo "=========================================="
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 -U --force-reinstall
    uv pip install triton  # For inductor backend
else
    # Clone and build PyTorch from the fix branch
    if [ ! -d "/root/pytorch" ]; then
        git clone --single-branch --branch vmoens/fix-conv-layout https://github.com/vmoens/pytorch.git
    fi
    cd /root/pytorch
    safe_git_pull main main
    # git fetch origin
    # git checkout vmoens/fix-conv-layout

    if [ "$NO_BUILD" = false ]; then
        # Thorough submodule initialization
        echo "Initializing submodules (this may take a while)..."
        git submodule sync --recursive
        git submodule update --init --recursive --force --jobs 8

        # Verify critical submodules exist
        if [ ! -f "/root/pytorch/third_party/gloo/CMakeLists.txt" ]; then
            echo "ERROR: gloo submodule not properly initialized, retrying..."
            git submodule update --init --recursive --force third_party/gloo
        fi

        # Install build dependencies (per PyTorch README)
        # IMPORTANT: numpy must be installed BEFORE building PyTorch for numpy support
        uv pip install numpy
        uv pip install cmake ninja pyyaml setuptools typing_extensions cffi future six requests dataclasses packaging
        uv pip install mkl-include mkl-static || true  # Optional but helps with BLAS

        # Install Triton for inductor backend
        uv pip install triton

        # ============================================================
        # CRITICAL: Install cuDNN before building PyTorch
        # ============================================================
        # Check if cuDNN is already installed
        if [ ! -f "/usr/local/cuda/include/cudnn.h" ] && [ ! -f "/usr/local/cuda/include/cudnn_version.h" ]; then
            echo "cuDNN not found, installing..."
            # Install cuDNN via apt (NVIDIA repos should be configured in the container)
            apt-get update
            apt-get install -y libcudnn9-dev libcudnn9-cuda-12 || {
                echo "apt install failed, trying alternative cuDNN installation..."
                # Alternative: download from NVIDIA (requires manual setup or use pip package)
                # For CUDA 12.x containers, cuDNN 9.x is recommended
                uv pip install nvidia-cudnn-cu12 || true
                # Link cuDNN headers/libs if installed via pip
                CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "")
                if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH/include" ]; then
                    echo "Linking cuDNN from pip package..."
                    ln -sf "$CUDNN_PATH/include/cudnn"*.h /usr/local/cuda/include/ 2>/dev/null || true
                    ln -sf "$CUDNN_PATH/lib/libcudnn"* /usr/local/cuda/lib64/ 2>/dev/null || true
                fi
            }
        fi

        # Verify cuDNN installation
        if [ -f "/usr/local/cuda/include/cudnn.h" ] || [ -f "/usr/local/cuda/include/cudnn_version.h" ]; then
            echo "cuDNN headers found at /usr/local/cuda/include/"
        else
            echo "ERROR: cuDNN headers still not found! PyTorch will be built WITHOUT cuDNN support."
            echo "This will cause CPU fallback for convolutions (very slow)."
            echo "Please install cuDNN manually before proceeding."
            # Don't exit - let user decide
        fi

        # Configure build for CUDA 13.0
        export CUDA_HOME=/usr/local/cuda
        export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
        export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
        export USE_CUDA=1
        export USE_CUDNN=1
        export USE_NCCL=1
        # Reduce parallel jobs if running out of memory (each job uses ~2GB RAM)
        export MAX_JOBS=64
        export CMAKE_BUILD_PARALLEL_LEVEL=16

        # Ensure CMake can find CUDA/cuDNN (per PyTorch build-from-source docs)
        export CMAKE_PREFIX_PATH="/usr/local/cuda:${CMAKE_PREFIX_PATH:-}"
        export CUDNN_INCLUDE_DIR="/usr/local/cuda/include"
        export CUDNN_LIB_DIR="/usr/local/cuda/lib64"

        # Build PyTorch from source (full output for live monitoring)
        echo "=========================================="
        echo "Building PyTorch from source..."
        echo "Using ccache: $(which ccache)"
        echo "MAX_JOBS: $MAX_JOBS"
        echo "CUDA_HOME: $CUDA_HOME"
        echo "=========================================="

        uv pip install -e . -v --no-build-isolation 2>&1
    else
        echo "Skipping PyTorch build (--no-build mode)"
    fi
fi

cd /root/

# Clone tensordict and rl if not present
if [ ! -d "/root/tensordict" ]; then
    git clone https://github.com/pytorch/tensordict
fi
if [ ! -d "/root/rl" ]; then
    git clone https://github.com/pytorch/rl
fi

# Pull tensordict
cd /root/tensordict
git fetch origin
git checkout main
safe_git_pull "main" "main"

# Pull rl - use ghstack checkout for SGLang tests or normal checkout
cd /root/rl
git fetch origin
if [ "$RUN_SGLANG_TESTS" = true ]; then
    echo "Checking out SGLang branch (gh/vmoens/214/head - includes all SGLang commits)..."
    # This is the ghstack branch for PR #3434: [Test] Add SGLang weight synchronization tests
    # It includes all SGLang commits including test_sglang.py and test_sglang_updaters.py
    git checkout -B sglang-tests origin/gh/vmoens/214/head
else
    git checkout fix-dreamer-2
    safe_git_pull "fix-dreamer-2" "fix-dreamer-2"
fi

if [ "$NO_BUILD" = false ]; then
    # Install other dependencies
    uv pip install pytest pytest-random-order ray gym gymnasium pillow

    cd /root/tensordict
    uv pip install -e .

    cd /root/rl
    uv pip install -e .
else
    echo "Skipping tensordict and rl builds (--no-build mode)"
fi

uv pip install mujoco dm_control wandb tqdm hydra-core moviepy

# Build torchvision from source only if NOT using nightly
# (nightly already includes torchvision)
if [ "$USE_NIGHTLY" = false ]; then
    if [ ! -d "/root/vision" ]; then
        git clone https://github.com/pytorch/vision.git /root/vision
    fi
    cd /root/vision
    git fetch origin
    git checkout main
    safe_git_pull "main" "main"

    if [ "$NO_BUILD" = false ]; then
        # Build torchvision against our custom PyTorch
        uv pip install pillow  # torchvision dependency
        uv pip install -e . -v --no-build-isolation --no-deps
    else
        echo "Skipping torchvision build (--no-build mode)"
    fi
fi

if [ "$NO_BUILD" = false ]; then
    apt-get update && apt-get install -y \
        libegl1 \
        libgl1 \
        libgles2 \
        libglvnd0
fi

export MUJOCO_GL=egl
export TORCHRL_WEIGHT_SYNC_TIMEOUT=120

echo "=========================================="
echo "Setup complete!"
echo "PyTorch build verification:"
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('cuDNN is_available:', torch.backends.cudnn.is_available())
print('cuDNN enabled:', torch.backends.cudnn.enabled)
if torch.backends.cudnn.is_available():
    print('cuDNN version:', torch.backends.cudnn.version())
else:
    print('ERROR: cuDNN NOT AVAILABLE! Convolutions will use slow CPU fallback.')
    print('This is likely because cuDNN was not installed before building PyTorch.')
print('NCCL available:', torch.distributed.is_nccl_available())

# Quick functional test
if torch.cuda.is_available() and torch.backends.cudnn.is_available():
    x = torch.randn(1, 3, 64, 64, device='cuda')
    conv = torch.nn.Conv2d(3, 32, 4, stride=2, device='cuda')
    y = conv(x)  # This should use cudnn_convolution
    print('Conv2d on CUDA: OK (should use cuDNN)')
"
echo "=========================================="

cd /root/rl

if [ "$BUILD_ONLY" = true ]; then
    echo "Build complete! Skipping dreamer execution (--build-only mode)"
    exit 0
fi

# ============================================================
# SGLang Tests Mode
# ============================================================
if [ "$RUN_SGLANG_TESTS" = true ]; then
    echo "=========================================="
    echo "Installing SGLang and LLM dependencies..."
    echo "=========================================="
    
    # Install SGLang and dependencies
    # Note: sglang[all] pulls an ancient vllm==0.1.2, so we install vllm separately
    uv pip install "sglang[all]" transformers accelerate
    # Install a modern vllm version (0.8.x is compatible with PyTorch 2.9+)
    uv pip install "vllm>=0.8.0" --upgrade
    uv pip install pytest pytest-timeout
    
    # Pre-download model to avoid timeout during tests
    echo "Pre-downloading Qwen/Qwen2.5-0.5B model..."
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')"
    
    echo "=========================================="
    echo "Pre-warming SGLang server (first run compiles CUDA kernels)..."
    echo "=========================================="
    
    # Pre-warm SGLang to compile CUDA kernels
    python -c "
import subprocess
import time
import requests
import sys

# Start SGLang server in background
print('Starting SGLang server for pre-warming...')
proc = subprocess.Popen(
    ['python3', '-m', 'sglang.launch_server',
     '--model-path', 'Qwen/Qwen2.5-0.5B',
     '--host', '127.0.0.1',
     '--port', '39999',
     '--tp-size', '1',
     '--mem-fraction-static', '0.3'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

# Wait for server to be ready (up to 10 minutes for first-time kernel compilation)
start = time.time()
timeout = 600
ready = False
while time.time() - start < timeout:
    try:
        resp = requests.get('http://127.0.0.1:39999/health', timeout=5)
        if resp.status_code == 200:
            ready = True
            break
    except:
        pass
    time.sleep(2)

elapsed = time.time() - start
if ready:
    print(f'SGLang server ready after {elapsed:.1f}s - kernels compiled and cached')
else:
    print(f'WARNING: SGLang server did not start within {timeout}s')
    proc.terminate()
    proc.wait(timeout=5)
    sys.exit(1)

# Shutdown server
proc.terminate()
proc.wait(timeout=10)
print('SGLang pre-warming complete')
"
    
    echo "=========================================="
    echo "Running SGLang tests..."
    echo "=========================================="
    
    cd /root/rl
    
    # Run SGLang tests with verbose output
    # --runslow runs the slow integration tests that require SGLang server
    python -m pytest test/llm/test_sglang.py test/llm/test_sglang_updaters.py -v -s --tb=short --runslow 2>&1 | tee sglang_tests.log
    
    echo "=========================================="
    echo "SGLang tests complete! Check sglang_tests.log for results."
    echo "=========================================="
    exit 0
fi

wandb login 51220607411b568888f5349f8647cbc92ed10407
# Run dreamer with inductor backend (should work now!)

# TORCH_LOGS="+dynamo" CUDA_VISIBLE_DEVICES=0 python sota-implementations/dreamer/dreamer.py optimization.autocast=false optimization.compile_backend=inductor 2>&1 | tee inductor_fixed.log
# TORCH_LOGS="+dynamo,recompiles" CUDA_VISIBLE_DEVICES=0 python sota-implementations/dreamer/dreamer.py optimization.autocast=false optimization.compile=true networks.use_scan=false 2>&1 | tee inductor_fixed.log
# python sota-implementations/dreamer/dreamer.py optimization.autocast=false networks.use_scan=false optimization.compile=false profiling.enabled=true 2>&1 | tee inductor_fixed.log
#TORCH_LOGS="+guards,recompiles" CUDA_VISIBLE_DEVICES=0 python sota-implementations/dreamer/dreamer.py optimization.autocast=false optimization.compile=false networks.rssm_rollout.compile=true networks.use_scan=false 2>&1 | tee inductor_fixed.log
#TORCH_LOGS="+dynamo,recompiles" CUDA_VISIBLE_DEVICES=0 python sota-implementations/dreamer/dreamer.py optimization.autocast=false optimization.compile=false networks.rssm_rollout.compile=true networks.use_scan=false profiling.enabled=true 2>&1 | tee inductor_fixed.log
#TORCH_LOGS="+guards,recompiles" CUDA_VISIBLE_DEVICES=0 \
# python sota-implementations/dreamer/dreamer.py \
#   optimization.autocast=false \
#   optimization.compile.enabled=true \
#   optimization.compile.losses=[actor] \
#   networks.rssm_rollout.compile=true \
#   networks.use_scan=false \
#   profiling.enabled=false \
#   profiling.collector.enabled=false \
#   logger.video=true \
# 2>&1 | tee inductor_fixed.log
python sota-implementations/dreamer/dreamer.py \
  optimization.autocast=false \
  optimization.compile.enabled=true \
  profiling.enabled=true \
  profiling.collector.enabled=true \
  logger.video=true \
2>&1 | tee inductor_fixed.log
