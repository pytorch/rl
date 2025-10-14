#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
apt-get update && apt-get install -y git wget curl gcc g++
apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev

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


## 3. Install mujoco
#printf "* Installing mujoco and related\n"
#mkdir -p $root_dir/.mujoco
#cd $root_dir/.mujoco/
##wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
##tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
##wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#wget https://www.roboti.us/download/mujoco200_linux.zip
#unzip mujoco200_linux.zip
## install mujoco-py locally
git clone https://github.com/vmoens/mujoco-py.git
cd mujoco-py
git checkout aws_fix2
mkdir -p mujoco_py/binaries/linux \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C mujoco_py/binaries/linux \
    && rm mujoco.tar.gz
wget https://pytorch.s3.amazonaws.com/torchrl/github-artifacts/mjkey.txt
cp mjkey.txt mujoco_py/binaries/
# Install poetry for mujoco-py build (it uses poetry as build backend)
uv pip install poetry
uv pip install -e . --no-build-isolation
cd ..

#cd $this_dir

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"


export MUJOCO_GL=egl
export MAX_IDLE_COUNT=1000 MUJOCO_GL=egl \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=:99 \
  PYOPENGL_PLATFORM=egl \
  LD_PRELOAD=$glew_path \
  NVIDIA_PATH=/usr/src/nvidia-470.63.01 \
  MUJOCO_PY_MJKEY_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/mjkey.txt \
  MUJOCO_PY_MUJOCO_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/linux/mujoco210 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pytorch/rl/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin
  TOKENIZERS_PARALLELISM=true
#  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/circleci/project/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin

# make env variables apparent

# uv pip install pip --upgrade

# Dependencies installed via uv pip (see converted script)
#conda install -c conda-forge fltk -y

# ROM licence for Atari
wget https://www.rarlab.com/rar/rarlinux-x64-5.7.1.tar.gz --no-check-certificate
tar -xzvf rarlinux-x64-5.7.1.tar.gz
mkdir Roms
wget http://www.atarimania.com/roms/Roms.rar
./rar/unrar e Roms.rar ./Roms -y
python -m atari_py.import_roms Roms
