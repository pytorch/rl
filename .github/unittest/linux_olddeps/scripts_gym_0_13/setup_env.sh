#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

# Make apt-get non-interactive
export DEBIAN_FRONTEND=noninteractive
# Pre-configure timezone data
ln -fs /usr/share/zoneinfo/UTC /etc/localtime
echo "UTC" > /etc/timezone

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Add NVIDIA repository for drivers
apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    ca-certificates

# Install basic build tools first
apt-get install -y vim git wget build-essential

# Install system libraries to fix version conflicts
apt-get install -y --no-install-recommends \
    libffi7 \
    libffi-dev \
    libtinfo6 \
    libtinfo-dev \
    libncurses5-dev \
    libncursesw5-dev

# Install OpenGL packages with focus on OSMesa
apt-get install -y --no-install-recommends \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    xvfb \
    mesa-utils \
    mesa-common-dev \
    libglu1-mesa-dev \
    libsdl2-dev \
    libsdl2-2.0-0 \
    pkg-config

if [ "${CU_VERSION:-}" == cpu ] ; then
  apt-get upgrade -y libstdc++6
  apt-get dist-upgrade -y
else
  apt-get install -y g++ gcc
fi

# Remove conflicting libraries from conda environment if they exist
rm -f "${env_dir}/lib/libtinfo.so"* || true
rm -f "${env_dir}/lib/libffi.so"* || true

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

# Use OSMesa for rendering
export MUJOCO_GL=osmesa
conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MUJOCO_GL=osmesa \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=:99 \
  PYOPENGL_PLATFORM=osmesa \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.6:/usr/lib/x86_64-linux-gnu/libGL.so \
  MUJOCO_PY_MJKEY_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/mjkey.txt \
  MUJOCO_PY_MUJOCO_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/linux/mujoco210 \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH:/home/circleci/project/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin \
  TOKENIZERS_PARALLELISM=true \
  PYGLET_GRAPHICS=opengl3

# make env variables apparent
conda deactivate && conda activate "${env_dir}"

#pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune
#conda install -c conda-forge fltk -y

# ROM licence for Atari
wget https://www.rarlab.com/rar/rarlinux-x64-5.7.1.tar.gz
tar -xzvf rarlinux-x64-5.7.1.tar.gz
mkdir Roms
wget http://www.atarimania.com/roms/Roms.rar
./rar/unrar e Roms.rar ./Roms -y
python -m atari_py.import_roms Roms
