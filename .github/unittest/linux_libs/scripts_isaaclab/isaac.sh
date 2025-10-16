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
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1

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

# Set LD_LIBRARY_PATH to prioritize conda environment libraries early
export LD_LIBRARY_PATH=${lib_dir}:${LD_LIBRARY_PATH:-}

# Ensure libexpat is at the correct version BEFORE installing other packages
conda install -c conda-forge expat -y

# Force the loader to pick conda's libexpat over the system one
if [ -f "${lib_dir}/libexpat.so.1" ]; then
  export LD_PRELOAD="${lib_dir}/libexpat.so.1:${LD_PRELOAD:-}"
elif [ -f "${lib_dir}/libexpat.so" ]; then
  export LD_PRELOAD="${lib_dir}/libexpat.so:${LD_PRELOAD:-}"
fi

# Quick diagnostic to confirm which expat is resolved by pyexpat
PYEXPAT_SO=$(python - <<'PY'
import importlib.util
spec = importlib.util.find_spec('pyexpat')
print(spec.origin)
PY
)
echo "* pyexpat module: ${PYEXPAT_SO}"
ldd "${PYEXPAT_SO}" | grep -i expat || true

# Reinstall Python to ensure it links against the correct expat
conda install --force-reinstall python=3.10 -y

# Pin pytorch to 2.5.1 for IsaacLab
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

python -m pip install --upgrade pip --disable-pip-version-check
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com --disable-pip-version-check
conda install conda-forge::"cmake>3.22" -y

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install sb3
cd ../

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  conda install "anaconda::cmake>=3.22" -y
  python -m pip install "pybind11[global]" --disable-pip-version-check
  python -m pip install git+https://github.com/pytorch/tensordict.git --disable-pip-version-check
else
  python -m pip install tensordict --disable-pip-version-check
fi

# smoke test
python -c "import tensordict"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation --disable-pip-version-check
python -c "import torchrl"

# Install pytest
python -m pip install pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures pytest-error-for-skips pytest-asyncio --disable-pip-version-check

# Run tests
python -m pytest test/test_libs.py -k isaac -s
