#!/usr/bin/env bash

set -e
set -v

export RELEASE=0
export TORCH_VERSION=nightly

set -euo pipefail
export PYTHON_VERSION="3.10"
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
apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"

cd "${root_dir}"

# The Isaac Lab Docker image already has isaacsim and isaaclab installed
# We just need to install tensordict and torchrl

# Check the existing Python environment
echo "* Checking existing Python environment:"
which python
python --version
python -c "import platform; print(f'Implementation: {platform.python_implementation()}')"

# Check if isaaclab is already available
echo "* Checking for existing isaaclab installation:"
python -c "import isaaclab; print(f'IsaacLab version: {isaaclab.__version__}')" || echo "WARNING: isaaclab not found"

# Install tensordict and torchrl
echo "* Installing tensordict from source..."
if [[ "$RELEASE" == 0 ]]; then
  python -m pip install "pybind11[global]" --disable-pip-version-check
  python -m pip install git+https://github.com/pytorch/tensordict.git --disable-pip-version-check
else
  python -m pip install tensordict --disable-pip-version-check
fi

# smoke test
python -c "import tensordict; print(f'TensorDict imported successfully')"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation --disable-pip-version-check
python -c "import torchrl; print(f'TorchRL imported successfully')"

# Install pytest
python -m pip install pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures pytest-error-for-skips pytest-asyncio --disable-pip-version-check

# Run tests
python -m pytest test/test_libs.py -k isaac -s
