#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

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
env_dir="${root_dir}/env"

cd "${root_dir}"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    printf "* Installing uv\n"
    # Try different Python commands
    if command -v python &> /dev/null; then
        python -m pip install uv
    else
        # Fallback to curl installation
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Create virtual environment using uv
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment with uv\n"
    uv venv "${env_dir}" --python "${PYTHON_VERSION}"
fi

# Activate the virtual environment
source "${env_dir}/bin/activate"

# Upgrade pip
uv pip install --upgrade pip

# Install dependencies from requirements.txt (we'll create this)
printf "* Installing dependencies (except PyTorch)\n"
if [ -f "${this_dir}/requirements.txt" ]; then
    uv pip install -r "${this_dir}/requirements.txt"
fi
