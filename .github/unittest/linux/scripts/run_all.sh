#!/usr/bin/env bash

set -euxo pipefail
set -v

# =============================================================================== #
# ================================ Init ========================================= #


if [[ $OSTYPE != 'darwin'* ]]; then
  apt-get update && apt-get upgrade -y
  apt-get install -y vim git wget curl cmake

  # Enable universe repository
  # apt-get install -y software-properties-common
  # add-apt-repository universe
  # apt-get update

  # apt-get install -y libsdl2-dev libsdl2-2.0-0

  apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev
  apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb

  if [ "${CU_VERSION:-}" == cpu ] ; then
    # solves version `GLIBCXX_3.4.29' not found for tensorboard
#    apt-get install -y gcc-4.9
    apt-get upgrade -y libstdc++6
    apt-get dist-upgrade -y
  else
    apt-get install -y g++ gcc cmake
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
env_dir="${root_dir}/.venv"

cd "${root_dir}"

# 1. Install uv
printf "* Installing uv\n"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create test environment at ./.venv
printf "python: ${PYTHON_VERSION}\n"
printf "* Creating a test environment with uv\n"
uv venv "${env_dir}" --python="${PYTHON_VERSION}"
source "${env_dir}/bin/activate"

# 3. Install dependencies (except PyTorch)
printf "* Installing dependencies (except PyTorch)\n"

if [ "${CU_VERSION:-}" == cpu ] ; then
  export MUJOCO_GL=glfw
else
  export MUJOCO_GL=egl
fi

export SDL_VIDEODRIVER=dummy

# Set environment variables
export MAX_IDLE_COUNT=1000
export PYOPENGL_PLATFORM=$MUJOCO_GL
export DISPLAY=:99
export LAZY_LEGACY_OP=False
export RL_LOGGING_LEVEL=DEBUG
export TOKENIZERS_PARALLELISM=true

# Install build dependencies FIRST (required for C++ extensions)
printf "* Installing build dependencies\n"
uv pip install setuptools wheel ninja "pybind11[global]"

# Install dependencies from requirements.txt
printf "* Installing dependencies from requirements.txt\n"
uv pip install -r "${this_dir}/requirements.txt"

# Install pip for compatibility with packages that expect it
uv pip install pip

echo "installing gymnasium"
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  uv pip install ale-py
  uv pip install sympy
  uv pip install "gymnasium[mujoco]>=1.1" mo-gymnasium[mujoco]
else
  uv pip install "gymnasium[atari,mujoco]>=1.1" mo-gymnasium[mujoco]
fi

# sanity check: remove?
python -c """
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
      uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -U
  else
      uv pip install torch torchvision --index-url https://download.pytorch.org/whl/$CU_VERSION -U
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
  uv pip install git+https://github.com/pytorch/tensordict.git
else
  uv pip install tensordict
fi

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation

if [ "${CU_VERSION:-}" != cpu ] ; then
  printf "* Installing VC1\n"
  python -c """
from torchrl.envs.transforms.vc1 import VC1Transform
VC1Transform.install_vc_models(auto_exit=True)
"""

  python -c """
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

Xvfb :99 -screen 0 1024x768x24 &

pytest test/smoke_test.py -v --durations 200
pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'
if [ "${CU_VERSION:-}" != cpu ] ; then
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda
else
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/test_distributed.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda
fi

coverage combine
coverage xml -i

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
