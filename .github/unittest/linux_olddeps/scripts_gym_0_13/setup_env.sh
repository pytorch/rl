#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

apt-get update && apt-get upgrade -y
printf "* Installing vim - git - wget - cmake\n"
apt-get install -y vim git wget cmake

printf "* Installing glfw - glew - osmesa part 1\n"
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libx11-dev \
  libegl-dev \
  librhash0 # For cmake

#printf "* Installing glfw - glew - osmesa part 2\n"
#apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0

if [ "${CU_VERSION:-}" == cpu ] ; then
  # solves version `GLIBCXX_3.4.29' not found for tensorboard
#    apt-get install -y gcc-4.9
  apt-get upgrade -y libstdc++6
  apt-get dist-upgrade -y
else
  apt-get install -y g++ gcc
fi

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

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
pip install -e .
cd ..

# OLD DM_CONTROL NOT SUPPORTED ANYMORE
# # Install dm_control
# git clone https://github.com/deepmind/dm_control
# cd dm_control
# git checkout c053360edea6170acfd9c8f65446703307d9d352
# pip install -e .
# cd ..

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

export MUJOCO_GL=egl
conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MUJOCO_GL=egl \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=unix:0.0 \
  PYOPENGL_PLATFORM=egl \
  NVIDIA_PATH=/usr/src/nvidia-470.63.01 \
  MUJOCO_PY_MJKEY_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/mjkey.txt \
  MUJOCO_PY_MUJOCO_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/linux/mujoco210 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/circleci/project/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin \
  TOKENIZERS_PARALLELISM=true

# make env variables apparent
conda deactivate && conda activate "${env_dir}"

#pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune
#conda install -c conda-forge fltk -y
