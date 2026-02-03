#!/bin/bash
set -e

# Parse command line arguments
NO_BUILD=false
USE_NIGHTLY=false
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
    esac
done

if [ "$NO_BUILD" = true ]; then
    echo "Running in --no-build mode: will pull repos but skip building"
fi
if [ "$USE_NIGHTLY" = true ]; then
    echo "Running in --nightly mode: will use PyTorch nightly wheels instead of building from source"
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
    apt-get update && apt-get install -y ccache
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
        git clone https://github.com/vmoens/pytorch.git
        cd pytorch
        git checkout vmoens/fix-conv-layout
    else
        cd pytorch
        git fetch origin
        git checkout vmoens/fix-conv-layout
        safe_git_pull "vmoens/fix-conv-layout" "vmoens/fix-conv-layout"
    fi

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

# Pull rl
cd /root/rl
git fetch origin
git checkout fix-dreamer
safe_git_pull "fix-dreamer" "fix-dreamer"

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

if [ "$NO_BUILD" = false ]; then
    uv pip install mujoco dm_control wandb tqdm hydra-core
fi

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
wandb login 51220607411b568888f5349f8647cbc92ed10407
# Run dreamer with inductor backend (should work now!)

# TORCH_LOGS="+dynamo" CUDA_VISIBLE_DEVICES=0 python sota-implementations/dreamer/dreamer.py optimization.autocast=false optimization.compile_backend=inductor 2>&1 | tee inductor_fixed.log
# TORCH_LOGS="+dynamo,recompiles" CUDA_VISIBLE_DEVICES=0 python sota-implementations/dreamer/dreamer.py optimization.autocast=false optimization.compile=false networks.use_scan=True 2>&1 | tee inductor_fixed.log
python sota-implementations/dreamer/dreamer.py optimization.autocast=false networks.use_scan=false optimization.compile=false profiling.enabled=true 2>&1 | tee inductor_fixed.log
