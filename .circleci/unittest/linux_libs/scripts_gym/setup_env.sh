#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
apt-get update && apt-get install -y git wget gcc g++

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
wget https://www.roboti.us/file/mjkey.txt
cp mjkey.txt mujoco_py/binaries/
pip install -e .
cd ..

#cd $this_dir

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

export MUJOCO_GL=egl
conda env config vars set \
  MUJOCO_GL=egl \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=unix:0.0 \
  PYOPENGL_PLATFORM=egl \
  LD_PRELOAD=$glew_path \
  NVIDIA_PATH=/usr/src/nvidia-470.63.01 \
  MUJOCO_PY_MJKEY_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/mjkey.txt \
  MUJOCO_PY_MUJOCO_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/linux/mujoco210 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/circleci/project/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin

#  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/src/nvidia-470.63.01 \

# make env variables apparent
conda deactivate && conda activate "${env_dir}"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune
#conda install -c conda-forge fltk -y

# ROM licence for Atari
wget https://www.rarlab.com/rar/rarlinux-x64-5.7.1.tar.gz --no-check-certificate
tar -xzvf rarlinux-x64-5.7.1.tar.gz
mkdir Roms
wget http://www.atarimania.com/roms/Roms.rar
./rar/unrar e Roms.rar ./Roms -y
python -m atari_py.import_roms Roms
