#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
#apt-get update && apt-get install -y git wget gcc g++ unzip

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

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

# set debug variables
conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MAGNUM_LOG=debug HABITAT_SIM_LOG=debug TOKENIZERS_PARALLELISM=true
conda deactivate && conda activate "${env_dir}"

pip3 install "cython<3"
conda install -c anaconda cython="<3.0.0" -y


# 3. Install git LFS
mkdir git_lfs
wget https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz --directory-prefix git_lfs
cd git_lfs
tar -xf git-lfs-linux-amd64-v2.9.0.tar.gz
chmod 755 install.sh
./install.sh
cd ..
git lfs install

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
echo "numpy<2.0" > constraints.txt
pip3 install -e habitat-lab --constraint constraints.txt
pip3 install -e habitat-baselines --constraint constraints.txt # install habitat_baselines
conda run python -m pip install "gym[atari,accept-rom-license]" pygame
