#!/usr/bin/env bash

set -e
set -v

#if [[ "${{ github.ref }}" =~ release/* ]]; then
#  export RELEASE=1
#  export TORCH_VERSION=stable
#else
export RELEASE=0
export TORCH_VERSION=nightly
#fi

set -euo pipefail
export PYTHON_VERSION="3.10"
export CU_VERSION="12.8"
export TAR_OPTIONS="--no-same-owner"
export UPLOAD_CHANNEL="nightly"
export TF_CPP_MIN_LOG_LEVEL=0
export BATCHED_PIPE_TIMEOUT=60
export TD_GET_DEFAULTS_TO_NONE=1
export OMNI_KIT_ACCEPT_EULA=yes

nvidia-smi

# Setup
apt-get update && apt-get install -y git wget gcc g++
apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

cd "${root_dir}"

# install conda
printf "* Installing conda\n"
wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash ./miniconda.sh -b -f -p "${conda_dir}"
eval "$(${conda_dir}/bin/conda shell.bash hook)"


conda create --prefix ${env_dir} python=3.10 -y
conda activate ${env_dir}

# Pin pytorch to 2.5.1 for IsaacLab
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

conda run -p ${env_dir} pip install --upgrade pip
conda run -p ${env_dir} pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
conda install conda-forge::"cmake>3.22" -y

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
conda run -p ${env_dir} ./isaaclab.sh --install sb3
cd ../

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  conda install "anaconda::cmake>=3.22" -y
  conda run -p ${env_dir} python3 -m pip install "pybind11[global]"
  conda run -p ${env_dir} python3 -m pip install git+https://github.com/pytorch/tensordict.git
else
  conda run -p ${env_dir} python3 -m pip install tensordict
fi

# smoke test
conda run -p ${env_dir} python -c "import tensordict"

printf "* Installing torchrl\n"
conda run -p ${env_dir} python setup.py develop
conda run -p ${env_dir} python -c "import torchrl"

# Install pytest
conda run -p ${env_dir} python -m pip install pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures pytest-error-for-skips pytest-asyncio

# Run tests
conda run -p ${env_dir} python -m pytest test/test_libs.py -k isaac -s
