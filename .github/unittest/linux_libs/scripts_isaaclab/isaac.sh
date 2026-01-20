#!/usr/bin/env bash

set -e
set -v

export RELEASE=0
export TORCH_VERSION=nightly

set -euo pipefail
export PYTHON_VERSION="3.11"
export CU_VERSION="12.8"
export TAR_OPTIONS="--no-same-owner"
export UPLOAD_CHANNEL="nightly"
export TF_CPP_MIN_LOG_LEVEL=0
export BATCHED_PIPE_TIMEOUT=60
export TD_GET_DEFAULTS_TO_NONE=1
export OMNI_KIT_ACCEPT_EULA=yes
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1

nvidia-smi

# Setup
apt-get update && apt-get install -y git wget gcc g++
apt-get install -y libglfw3 libgl1 libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"

cd "${root_dir}"

# The Isaac Lab Docker image has its Python environment in /workspace/isaaclab
# We need to use the isaaclab.sh wrapper to access the correct Python
ISAACLAB_DIR="/workspace/isaaclab"
ISAACLAB_PYTHON="${ISAACLAB_DIR}/isaaclab.sh"

# Check the existing Python environment
echo "* Checking IsaacLab environment:"
ls -la "${ISAACLAB_DIR}" || echo "WARNING: ${ISAACLAB_DIR} not found"

# Check if isaaclab.sh exists
if [[ -f "${ISAACLAB_PYTHON}" ]]; then
    echo "* Using IsaacLab's Python environment via isaaclab.sh"
    
    # Test isaaclab import
    echo "* Checking for existing isaaclab installation:"
    "${ISAACLAB_PYTHON}" -p -c "import isaaclab; print(f'IsaacLab found')" || echo "WARNING: isaaclab import failed"
    
    # Install tensordict with --no-deps to avoid packaging conflicts
    # Isaac Lab already has torch, numpy, etc. installed
    echo "* Installing tensordict from source (no-deps to avoid packaging conflicts)..."
    if [[ "$RELEASE" == 0 ]]; then
        "${ISAACLAB_PYTHON}" -p -m pip install "pybind11[global]" --disable-pip-version-check
        "${ISAACLAB_PYTHON}" -p -m pip install git+https://github.com/pytorch/tensordict.git --no-deps --disable-pip-version-check
        # Install only the missing dependencies that won't conflict
        "${ISAACLAB_PYTHON}" -p -m pip install cloudpickle orjson pyvers --disable-pip-version-check
    else
        "${ISAACLAB_PYTHON}" -p -m pip install tensordict --no-deps --disable-pip-version-check
        "${ISAACLAB_PYTHON}" -p -m pip install cloudpickle orjson pyvers --disable-pip-version-check
    fi
    
    # smoke test
    "${ISAACLAB_PYTHON}" -p -c "import tensordict; print(f'TensorDict imported successfully')"
    
    # Install torchrl with --no-deps to avoid conflicts, then install missing deps
    printf "* Installing torchrl\n"
    "${ISAACLAB_PYTHON}" -p -m pip install -e "${root_dir}" --no-build-isolation --no-deps --disable-pip-version-check
    # Install torchrl dependencies that are likely missing (ray is optional for tests)
    "${ISAACLAB_PYTHON}" -p -m pip install ray --disable-pip-version-check || echo "WARNING: ray installation failed (optional)"
    "${ISAACLAB_PYTHON}" -p -c "import torchrl; print(f'TorchRL imported successfully')"
    
    # Install pytest
    "${ISAACLAB_PYTHON}" -p -m pip install pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures pytest-error-for-skips pytest-asyncio --disable-pip-version-check
    
    # Run tests
    cd "${root_dir}"
    "${ISAACLAB_PYTHON}" -p -m pytest test/test_libs.py -k isaac -s
else
    echo "ERROR: Could not find isaaclab.sh at ${ISAACLAB_PYTHON}"
    echo "* Listing /workspace contents:"
    ls -la /workspace || echo "WARNING: /workspace not found"
    exit 1
fi
