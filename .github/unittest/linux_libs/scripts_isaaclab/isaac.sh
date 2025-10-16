#!/usr/bin/env bash

set -e
set -v

#if [[ "${{ github.ref }}" =~ release/* ]]; then
#  export RELEASE=1
#  export TORCH_VERSION=stable
#else
export RELEASE=0
export TORCH_VERSION=nightly
#fi

set -euo pipefail
export PYTHON_VERSION="3.10"
export CU_VERSION="12.8"
export TAR_OPTIONS="--no-same-owner"
export UPLOAD_CHANNEL="nightly"
export TF_CPP_MIN_LOG_LEVEL=0
export BATCHED_PIPE_TIMEOUT=60
export TD_GET_DEFAULTS_TO_NONE=1
export OMNI_KIT_ACCEPT_EULA=yes

nvidia-smi

# Setup
apt-get update && apt-get install -y git wget curl gcc g++
apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/.venv"

cd "${root_dir}"

# Install uv
printf "* Installing uv\n"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment with uv
printf "* Creating a test environment with uv\n"
uv venv "${env_dir}" --python="3.10"
source "${env_dir}/bin/activate"

# Pin pytorch to 2.5.1 for IsaacLab
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
uv pip install "cmake>3.22"

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install sb3
cd ../

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv pip install "pybind11[global]"
  uv pip install git+https://github.com/pytorch/tensordict.git
else
  uv pip install tensordict
fi

# smoke test
python -c "import tensordict"


printf "* Installing build dependencies\n"
uv pip install setuptools wheel ninja "pybind11[global]"

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation
python -c "import torchrl"

# Install pytest
uv pip install pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures pytest-error-for-skips pytest-asyncio

# Run tests
python -m pytest test/test_libs.py -k isaac -s
