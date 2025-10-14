#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
#apt-get update && apt-get install -y git wget curl gcc g++ unzip

git config --global --add safe.directory '*'
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

# set debug variables
export MAX_IDLE_COUNT=1000
export MAGNUM_LOG=debug
export HABITAT_SIM_LOG=debug
export TOKENIZERS_PARALLELISM=true

uv pip install "cython<3"


# 3. Install git LFS
mkdir git_lfs
wget https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz --directory-prefix git_lfs
cd git_lfs
tar -xf git-lfs-linux-amd64-v2.9.0.tar.gz
chmod 755 install.sh
./install.sh
cd ..
git lfs install

# 4. Install dependencies
printf "* Installing dependencies (except PyTorch)\n"

uv pip install pytest pytest-cov pytest-rerunfailures pytest-mock pytest-instafail \
  pybind11 scipy expecttest hydra-core pytest-timeout "moviepy<2.0.0" "gym[atari,accept-rom-license]" \
  pygame

# Install habitat-sim using pip (conda package not available with uv)
uv pip install habitat-sim

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
uv pip install -e habitat-lab
uv pip install -e habitat-baselines  # install habitat_baselines
