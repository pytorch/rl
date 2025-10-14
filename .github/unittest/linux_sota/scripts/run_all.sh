#!/usr/bin/env bash

set -euxo pipefail
set -v

# ==================================================================================== #
# ================================ Init ============================================== #


apt-get update && apt-get upgrade -y
apt-get install -y vim git wget curl cmake

apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libosmesa6-dev
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2
apt-get install -y g++ gcc patchelf

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# from cudagl docker image
cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


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

# 3. Install mujoco
printf "* Installing mujoco and related\n"
mkdir -p $root_dir/.mujoco
cd $root_dir/.mujoco/
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
cd "${root_dir}"

# 4. Install dependencies (except PyTorch)
printf "* Installing dependencies (except PyTorch)\n"

export MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LAZY_LEGACY_OP=False
export COMPOSITE_LP_AGGREGATE=0
export MAX_IDLE_COUNT=1000
export DISPLAY=:99
export BATCHED_PIPE_TIMEOUT=60
export TOKENIZERS_PARALLELISM=true

uv pip install hypothesis future cloudpickle pytest pytest-cov pytest-mock \
  pytest-instafail pytest-rerunfailures pytest-timeout pytest-asyncio \
  expecttest "pybind11[global]" pyyaml scipy hydra-core wandb \
  tensorboard mlflow submitit

# Install pip for compatibility with packages that expect it
uv pip install pip

# install d4rl
# Install poetry first (free-mujoco-py needs it as build backend)
uv pip install poetry
uv pip install free-mujoco-py
uv pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# compile mujoco-py (bc it's done at runtime for whatever reason someone thought it was a good idea)
python -c """import gym;import d4rl"""

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #


if [[ ${#CU_VERSION} -eq 4 ]]; then
    CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
elif [[ ${#CU_VERSION} -eq 5 ]]; then
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
fi
echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"

# submodules
git submodule sync && git submodule update --init --recursive

uv pip install ale-py -U
uv pip install "gym[atari,accept-rom-license]" "gymnasium>=1.1.0" -U

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install --pre torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      uv pip install --pre torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu
  else
      uv pip install torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# smoke test
python -c "import functorch"

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv pip install git+https://github.com/pytorch/tensordict.git
else
  uv pip install tensordict
fi

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation

# ==================================================================================== #
# ================================ Run tests ========================================= #


bash ${this_dir}/run_test.sh

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
