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

export MUJOCO_GL=egl
conda env config vars set \
  MUJOCO_GL=egl \
  SDL_VIDEODRIVER=dummy \
  DISPLAY=unix:0.0 \
  PYOPENGL_PLATFORM=egl \
  NVIDIA_PATH=/usr/src/nvidia-470.63.01 \
  sim_backend=MUJOCO \
  LAZY_LEGACY_OP=False

# make env variables apparent
conda deactivate && conda activate "${env_dir}"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda install conda-forge::ffmpeg -y

pip install robohive
# make sure only gymnasium is available
pip uninstall gym -y
