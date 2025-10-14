#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -euxo pipefail

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
printf "* Installing uv\n"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create test environment at ./.venv
printf "python: ${PYTHON_VERSION}\n"
printf "* Creating a test environment with uv\n"
uv venv "${env_dir}" --python="${PYTHON_VERSION}"
source "${env_dir}/bin/activate"

# 3. Install dependencies (except PyTorch)
printf "* Installing dependencies (except PyTorch)\n"

uv pip install hypothesis future cloudpickle pytest pytest-cov pytest-mock \
  pytest-instafail pytest-rerunfailures pytest-error-for-skips pytest-asyncio \
  expecttest "pybind11[global]" pyyaml scipy hydra-core "jax[cuda12]>=0.7.0" \
  brax psutil

#yum makecache
# sudo yum -y install glfw
#yum -y install glfw-devel
#yum -y install libGLEW
#yum -y install gcc-c++
