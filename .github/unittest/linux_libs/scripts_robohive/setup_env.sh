#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
apt-get update && apt-get install -y git wget curl gcc g++ cmake

git config --global --add safe.directory '*'
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


#git clone https://github.com/vmoens/mujoco-py.git
#cd mujoco-py
#git checkout aws_fix2
#mkdir -p mujoco_py/binaries/linux \
#    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#    && tar -xf mujoco.tar.gz -C mujoco_py/binaries/linux \
#    && rm mujoco.tar.gz
#wget https://pytorch.s3.amazonaws.com/torchrl/github-artifacts/mjkey.txt
#cp mjkey.txt mujoco_py/binaries/
#uv pip install -e . --no-build-isolation
#cd ..

#cd $this_dir

# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

export MUJOCO_GL=egl
export MAX_IDLE_COUNT=1000 MUJOCO_GL=egl \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=:99 \
  PYOPENGL_PLATFORM=egl \
  NVIDIA_PATH=/usr/src/nvidia-470.63.01 \
  sim_backend=MUJOCO \
  LAZY_LEGACY_OP=False \
  TOKENIZERS_PARALLELISM=true

# make env variables apparent

uv pip install pip --upgrade

# Dependencies installed via uv pip (see converted script)

apt-get install -y ffmpeg

uv pip install robohive

python -m robohive_init

# make sure only gymnasium is available
# pip uninstall gym -y
