#!/bin/bash
set -e

# Parse command line arguments
NO_BUILD=false
USE_NIGHTLY=false
BUILD_ONLY=false
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

cd /root/

# ============================================================
# GitHub Credentials Setup (required for private repos like prof)
# ============================================================
echo "=========================================="
echo "Checking GitHub credentials..."
echo "=========================================="

if [ -z "$GH_TOKEN" ]; then
    echo "ERROR: GH_TOKEN environment variable is not set!"
    echo "The prof repo (github.com/vmoens/prof) is private and requires authentication."
    echo ""
    echo "To fix this, set your GitHub Personal Access Token before running:"
    echo "  export GH_TOKEN=your_github_token"
    echo ""
    echo "You can create a token at: https://github.com/settings/tokens"
    echo "Required scopes: repo (full control of private repositories)"
    exit 1
fi

# Configure git credentials
git config --global credential.helper store
echo "https://vmoens:${GH_TOKEN}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials
echo "Git credentials configured for vmoens"

# Test credentials by checking access to the private prof repo
echo "Testing GitHub credentials with prof repo..."
if curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token ${GH_TOKEN}" \
    "https://api.github.com/repos/vmoens/prof" | grep -q "200"; then
    echo "GitHub credentials verified successfully!"
else
    echo "WARNING: Could not verify access to vmoens/prof repo."
    echo "The token may not have the required permissions, or the repo doesn't exist."
    echo "Continuing anyway - git clone will fail if credentials are invalid."
fi
echo "=========================================="

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

# Clone tensordict, rl, and prof if not present
if [ ! -d "/root/tensordict" ]; then
    git clone https://github.com/pytorch/tensordict
fi
if [ ! -d "/root/rl" ]; then
    git clone https://github.com/pytorch/rl
fi
if [ ! -d "/root/prof" ]; then
    git clone https://github.com/vmoens/prof
fi

# Pull tensordict
echo "=========================================="
echo "Updating tensordict repo..."
echo "=========================================="
cd /root/tensordict
git fetch origin
git checkout main
safe_git_pull "main" "main"
echo "tensordict: $(git rev-parse --short HEAD) on $(git branch --show-current)"

# Pull rl
echo "=========================================="
echo "Updating rl repo..."
echo "=========================================="
cd /root/rl
git fetch origin
git checkout dreamer-profile
safe_git_pull "dreamer-profile" "dreamer-profile"
echo "rl: $(git rev-parse --short HEAD) on $(git branch --show-current)"

# Pull prof
echo "=========================================="
echo "Updating prof repo..."
echo "=========================================="
cd /root/prof
git fetch origin
git checkout mp-backend
safe_git_pull "mp-backend" "mp-backend"
echo "prof: $(git rev-parse --short HEAD) on $(git branch --show-current 2>/dev/null || echo 'detached')"

if [ "$NO_BUILD" = false ]; then
    # Install other dependencies
    echo "=========================================="
    echo "Installing dependencies..."
    echo "=========================================="
    uv pip install pytest pytest-random-order ray gym gymnasium pillow

    echo "=========================================="
    echo "Installing tensordict..."
    echo "=========================================="
    cd /root/tensordict
    uv pip install -e .

    echo "=========================================="
    echo "Installing rl (torchrl)..."
    echo "=========================================="
    cd /root/rl
    uv pip install -e .

    echo "=========================================="
    echo "Installing prof for distributed profiling..."
    echo "=========================================="
    cd /root/prof
    # Prof uses uv_build as build backend, need to install it first
    uv pip install uv_build
    uv pip install -e .
    # Verify prof installed correctly
    python -c "import prof; print('prof version check - prepare:', hasattr(prof, 'prepare'))" || echo "Warning: prof import check failed"
else
    echo "Skipping tensordict, rl, and prof builds (--no-build mode)"
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
# Create traces directory for prof output
mkdir -p /root/traces

echo "=========================================="
echo "Running Dreamer with distributed profiling"
echo "=========================================="

# Verify prof is correctly installed before running
echo "Verifying prof installation..."
python -c "
import prof
print('prof module location:', prof.__file__)
print('prof.prepare available:', hasattr(prof, 'prepare'))
print('prof.profile available:', hasattr(prof, 'profile'))
print('prof.PROF_ENABLED:', prof.PROF_ENABLED)
" || {
    echo "ERROR: prof package not properly installed!"
    echo "Attempting to debug..."
    python -c "import sys; print('sys.path:', sys.path)"
    python -c "import prof; print(dir(prof))" || echo "Cannot import prof at all"
    exit 1
}

echo ""
echo "Prof configuration:"
echo "  PROF_ENABLED=1"
echo "  PROF_ITERATIONS=50-55"
echo "  PROF_OUTPUT_DIR=/root/traces"
echo "  PROF_MODE=lite"
echo ""
echo "Dreamer configuration:"
echo "  optimization.compile.enabled=true"
echo "  profiling.enabled=true"
echo "  profiling.distributed.enabled=true"
echo "=========================================="

# Run dreamer with distributed profiling via prof
# PROF_ENABLED=1: Enable prof distributed profiling
# PROF_ITERATIONS: Profile iterations 50-55 (after compile warmup, around weight update)
# PROF_OUTPUT_DIR: Where to save trace files
# PROF_MODE=lite: Only trace explicit regions (not all PyTorch ops)
PROF_ENABLED=1 \
PROF_ITERATIONS=50-55 \
PROF_OUTPUT_DIR=/root/traces \
PROF_MODE=lite \
python sota-implementations/dreamer/dreamer.py \
  optimization.autocast=true \
  optimization.compile.enabled=false \
  profiling.enabled=true \
  profiling.distributed.enabled=true \
  profiling.collector.enabled=false \
  logger.video=false \
2>&1 | tee dreamer_profiled.log

# Merge traces after run completes
echo "=========================================="
echo "Merging trace files..."
echo "=========================================="
echo "Listing trace files in /root/traces/:"
ls -la /root/traces/ || echo "Directory does not exist"

if ls /root/traces/trace_*.json 1> /dev/null 2>&1; then
    # Use prof-merge entry point (installed via pip install -e prof)
    prof-merge /root/traces/trace_*.json -o /root/traces/merged_trace.json
    echo "Merged trace saved to /root/traces/merged_trace.json"
    echo "To view: python /root/prof/resources/open_trace_in_ui /root/traces/merged_trace.json"
else
    echo "No trace files found in /root/traces/"
fi
