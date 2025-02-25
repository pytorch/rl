#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
apt-get update && apt-get upgrade -y && apt-get install -y git
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
apt-get install -y wget \
    gcc \
    g++ \
    unzip \
    curl \
    patchelf \
    libosmesa6 \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    swig3.0 \
    libglew-dev \
    libx11-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libsdl2-dev \
    libsdl2-2.0-0 \
    x11proto-dev \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

# Upgrade specific package
apt-get upgrade -y libstdc++6

cd /usr/lib/x86_64-linux-gnu
ln -s libglut.so.3.12 libglut.so.3
cd $this_dir

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

set -e

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

# 2. Create test environment at ./env
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
conda activate "${env_dir}"

#git clone https://github.com/vmoens/mujoco-py.git
#cd mujoco-py
#git checkout aws_fix2
#mkdir -p mujoco_py/binaries/linux \
#    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#    && tar -xf mujoco.tar.gz -C mujoco_py/binaries/linux \
#    && rm mujoco.tar.gz
#wget https://pytorch.s3.amazonaws.com/torchrl/github-artifacts/mjkey.txt
#cp mjkey.txt mujoco_py/binaries/
#pip install -e .
#cd ..

#cd $this_dir

# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MUJOCO_GL=egl \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=unix:0.0 \
  PYOPENGL_PLATFORM=egl \
  NVIDIA_PATH=/usr/src/nvidia-470.63.01 \
  sim_backend=MUJOCO \
  LAZY_LEGACY_OP=False \
  TOKENIZERS_PARALLELISM=true

# make env variables apparent
conda deactivate && conda activate "${env_dir}"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda install conda-forge::ffmpeg -y

pip install robohive

python3 -m robohive_init

# make sure only gymnasium is available
# pip uninstall gym -y
