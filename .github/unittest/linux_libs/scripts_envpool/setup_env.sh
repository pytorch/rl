#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
apt-get update && apt-get upgrade -y && apt-get install -y git cmake
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
apt-get install -y wget curl \
    gcc \
    g++ \
    unzip \
    curl \
    patchelf \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    swig3.0 \
    libglew-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2


# Upgrade specific package
apt-get upgrade -y libstdc++6

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/.venv"

cd "${root_dir}"


# 1. Install uv
printf "* Installing uv
"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"


# 2. Create test environment at ./.venv
printf "python: ${PYTHON_VERSION}
"
printf "* Creating a test environment with uv
"
uv venv "${env_dir}" --python="${PYTHON_VERSION}"
source "${env_dir}/bin/activate"


# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

uv pip install pip --upgrade

# Dependencies installed via uv pip (see converted script)

apt update
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 MUJOCO_GL=egl


if [[ $OSTYPE != 'darwin'* ]]; then
  # install ale-py: manylinux names are broken for CentOS so we need to manually download and
  # rename them
  PY_VERSION=$(python --version)
  echo "installing ale-py for ${PY_PY_VERSION}"
  if [[ $PY_VERSION == *"3.9"* ]]; then
    wget https://files.pythonhosted.org/packages/a0/98/4316c1cedd9934f9a91b6e27a9be126043b4445594b40cfa391c8de2e5e8/ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    uv pip install ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.10"* ]]; then
    wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    mv ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    uv pip install ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.11"* ]]; then
    wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    mv ale_py-0.8.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    uv pip install ale_py-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  fi
  echo "installing gym"
  # envpool does not currently work with gymnasium
  uv pip install "gym[atari,accept-rom-license]<1.0"
else
  uv pip install "gym[atari,accept-rom-license]<1.0"
fi
uv pip install envpool treevalue

# wget https://files.pythonhosted.org/packages/a1/ee/b35ab21e71d34cdcedf05c0d32abca3a3e87d669bdc772660165ee2f55b8/envpool-0.8.2-cp39-cp39-manylinux_2_24_x86_64.whl
# uv pip install envpool-0.8.2-cp39-cp39-manylinux_2_24_x86_64.whl
# rm envpool-0.8.2-cp39-cp39-manylinux_2_24_x86_64.whl
