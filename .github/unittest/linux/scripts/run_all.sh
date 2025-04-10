#!/usr/bin/env bash

set -euxo pipefail
set -v

# =============================================================================== #
# ================================ Init ========================================= #


if [[ $OSTYPE != 'darwin'* ]]; then
  apt-get update && apt-get upgrade -y
  apt-get install -y vim git wget libsdl2-dev libsdl2-2.0-0

  apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev
  apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2

  if [ "${CU_VERSION:-}" == cpu ] ; then
    # solves version `GLIBCXX_3.4.29' not found for tensorboard
#    apt-get install -y gcc-4.9
    apt-get upgrade -y libstdc++6
    apt-get dist-upgrade -y
  else
    apt-get install -y g++ gcc
  fi

fi

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $OSTYPE != 'darwin'* ]]; then
  # from cudagl docker image
  cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi


# ==================================================================================== #
# ================================ Setup env ========================================= #

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

if [ "${CU_VERSION:-}" == cpu ] ; then
  export MUJOCO_GL=glfw
else
  export MUJOCO_GL=egl
fi

export DISPLAY=:0
export SDL_VIDEODRIVER=dummy

# legacy from bash scripts: remove?
conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$MUJOCO_GL DISPLAY=:0 SDL_VIDEODRIVER=dummy LAZY_LEGACY_OP=False RL_LOGGING_LEVEL=DEBUG TOKENIZERS_PARALLELISM=true

pip3 install pip --upgrade
pip install virtualenv

conda env update --file "${this_dir}/environment.yml" --prune

# Reset conda env variables
conda deactivate
conda activate "${env_dir}"

echo "installing gymnasium"
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  pip3 install ale-py
  pip3 install sympy
  pip3 install "gymnasium[mujoco]>=1.1" mo-gymnasium[mujoco]
else
  pip3 install "gymnasium[atari,mujoco]>=1.1" mo-gymnasium[mujoco]
fi
pip3 install "mujoco" -U

# sanity check: remove?
python3 -c """
import dm_control
from dm_control import composer
from tensorboard import *
from google.protobuf import descriptor as _descriptor
"""

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #

unset PYTORCH_VERSION

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
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu -U
  else
      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/$CU_VERSION -U
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# smoke test
python -c "import functorch"

## install snapshot
#if [[ "$TORCH_VERSION" == "nightly" ]]; then
#  pip3 install git+https://github.com/pytorch/torchsnapshot
#else
#  pip3 install torchsnapshot
#fi

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  pip3 install git+https://github.com/pytorch/tensordict.git
else
  pip3 install tensordict
fi

printf "* Installing torchrl\n"
python setup.py develop


if [ "${CU_VERSION:-}" != cpu ] ; then
  printf "* Installing VC1\n"
  python3 -c """
from torchrl.envs.transforms.vc1 import VC1Transform
VC1Transform.install_vc_models(auto_exit=True)
"""

  python3 -c """
import vc_models
from vc_models.models.vit import model_utils
print(model_utils)
"""
fi

# ==================================================================================== #
# ================================ Run tests ========================================= #


export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
## Avoid error: "fatal: unsafe repository"
#git config --global --add safe.directory '*'
#root_dir="$(git rev-parse --show-toplevel)"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=100
export BATCHED_PIPE_TIMEOUT=60

pytest test/smoke_test.py -v --durations 200
pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'
if [ "${CU_VERSION:-}" != cpu ] ; then
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --timeout=120 --mp_fork_if_no_cuda
else
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/test_distributed.py \
    --timeout=120 --mp_fork_if_no_cuda
fi

coverage combine
coverage xml -i

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
