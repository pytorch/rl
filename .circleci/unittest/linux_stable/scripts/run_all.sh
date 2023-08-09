#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.


## setup_env.sh
set -euxo pipefail

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

## 3. Install mujoco
#printf "* Installing mujoco and related\n"
#mkdir -p $root_dir/.mujoco
#cd $root_dir/.mujoco/
#wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
#tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
#wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#tar -xf mujoco210-linux-x86_64.tar.gz
#cd $this_dir

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

PRIVATE_MUJOCO_GL=osmesa

export MUJOCO_GL=$PRIVATE_MUJOCO_GL
echo "MUJOCO_GL"
echo $MUJOCO_GL
conda env config vars set MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210 \
  DISPLAY=unix:0.0 \
  MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin \
  SDL_VIDEODRIVER=dummy \
  MUJOCO_GL=$PRIVATE_MUJOCO_GL \
  PYOPENGL_PLATFORM=$PRIVATE_MUJOCO_GL

# Software rendering requires GLX and OSMesa.
if [ $PRIVATE_MUJOCO_GL == 'egl' ] || [ $PRIVATE_MUJOCO_GL == 'osmesa' ] ; then
  yum install -y glew
  yum install -y glfw-devel
  yum install -y mesa-libGL
  yum install -y mesa-libGL-devel
  yum install -y mesa-libOSMesa-devel
  yum -y install egl-utils
  yum -y install freeglut
fi

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

if [[ $OSTYPE != 'darwin'* ]]; then
  PY_VERSION=$(python --version)
  echo "installing ale-py for ${PY_VERSION}"
  pip install ale-py
  echo "installing gymnasium"
  pip install "gymnasium[atari,accept-rom-license]"
  pip install mo-gymnasium[mujoco]  # requires here bc needs mujoco-py
else
  pip install "gymnasium[atari,accept-rom-license]"
  pip install mo-gymnasium[mujoco]  # requires here bc needs mujoco-py
fi

## Install.sh
unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

if [ "${CU_VERSION:-}" == cpu ] ; then
    version="cpu"
    echo "Using cpu build"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
fi

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [ "${CU_VERSION:-}" == cpu ] ; then
    pip3 install --progress-bar=off torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
else
    pip3 install --progress-bar=off torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
fi

# install tensordict
pip install --progress-bar=off git+https://github.com/pytorch-labs/tensordict.git

# smoke test
python -c "import torch;import functorch"

printf "* Installing torchrl\n"
printf "g++ version: "
gcc --version

python setup.py develop

## Run_test.sh
pip install glfw

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch

python -c "import ale_py; import shimmy; import gymnasium; print(gymnasium.envs.registration.registry.keys())"

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest --instafail -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py
coverage combine
coverage xml -i
