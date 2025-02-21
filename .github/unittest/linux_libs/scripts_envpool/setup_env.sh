#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
apt-get update && apt-get install -y \
    git \
    wget \
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

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
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

# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

apt update
apt-get install libgl1-mesa-glx

conda deactivate
conda activate "${env_dir}"

if [[ $OSTYPE != 'darwin'* ]]; then
  # install ale-py: manylinux names are broken for CentOS so we need to manually download and
  # rename them
  PY_VERSION=$(python --version)
  echo "installing ale-py for ${PY_PY_VERSION}"
  if [[ $PY_VERSION == *"3.7"* ]]; then
    wget https://files.pythonhosted.org/packages/ab/fd/6615982d9460df7f476cad265af1378057eee9daaa8e0026de4cedbaffbd/ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.8"* ]]; then
    wget https://files.pythonhosted.org/packages/0f/8a/feed20571a697588bc4bfef05d6a487429c84f31406a52f8af295a0346a2/ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.9"* ]]; then
    wget https://files.pythonhosted.org/packages/a0/98/4316c1cedd9934f9a91b6e27a9be126043b4445594b40cfa391c8de2e5e8/ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.10"* ]]; then
    wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    mv ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.11"* ]]; then
    wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    mv ale_py-0.8.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  fi
  echo "installing gym"
  # envpool does not currently work with gymnasium
  pip install "gym[atari,accept-rom-license]<1.0"
else
  pip install "gym[atari,accept-rom-license]<1.0"
fi
pip install envpool treevalue

# wget https://files.pythonhosted.org/packages/a1/ee/b35ab21e71d34cdcedf05c0d32abca3a3e87d669bdc772660165ee2f55b8/envpool-0.8.2-cp39-cp39-manylinux_2_24_x86_64.whl
# pip install envpool-0.8.2-cp39-cp39-manylinux_2_24_x86_64.whl
# rm envpool-0.8.2-cp39-cp39-manylinux_2_24_x86_64.whl
