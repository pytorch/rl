#!/usr/bin/env bash

set -euxo pipefail
set -v

# =============================================================================== #
# ================================ Init ========================================= #

if [[ $OSTYPE != 'darwin'* ]]; then
  # Prevent interactive prompts (notably tzdata) in CI.
  export DEBIAN_FRONTEND=noninteractive
  export TZ="${TZ:-Etc/UTC}"
  ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
  echo "${TZ}" > /etc/timezone || true

  apt-get update
  apt-get install -y --no-install-recommends tzdata
  dpkg-reconfigure -f noninteractive tzdata || true

  apt-get upgrade -y
  apt-get install -y vim git wget cmake curl python3-dev

  # SDL2 and freetype needed for building pygame from source (Python 3.14+)
  apt-get install -y libsdl2-dev libsdl2-2.0-0 libsdl2-mixer-dev libsdl2-image-dev libsdl2-ttf-dev
  apt-get install -y libfreetype6-dev pkg-config

  apt-get install -y libglfw3 libosmesa6 libglew-dev
  apt-get install -y libglvnd0 libgl1 libglx0 libglx-mesa0 libegl1 libgles2 xvfb

  if [ "${CU_VERSION:-}" == cpu ] ; then
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
env_dir="${root_dir}/venv"

cd "${root_dir}"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with uv
printf "* Creating venv with Python ${PYTHON_VERSION}\n"
# IMPORTANT: ensure a clean environment.
# In CI (and some local workflows), the workspace directory can be reused across runs.
# A reused venv may contain packages that violate our constraints (e.g. transformers'
# huggingface-hub upper bound), and `uv pip install` does not always guarantee
# downgrades of already-present packages unless the environment is clean.
rm -rf "${env_dir}"
uv venv --python "${PYTHON_VERSION}" "${env_dir}"
source "${env_dir}/bin/activate"
uv_pip_install() {
  uv pip install --no-progress --python "${env_dir}/bin/python" "$@"
}

# Verify CPython
python -c "import sys; assert sys.implementation.name == 'cpython', f'Expected CPython, got {sys.implementation.name}'"

# Set environment variables
if [ "${CU_VERSION:-}" == cpu ] ; then
  export MUJOCO_GL=glfw
else
  export MUJOCO_GL=egl
fi

export SDL_VIDEODRIVER=dummy
export PYOPENGL_PLATFORM=$MUJOCO_GL
export DISPLAY=:99
export LAZY_LEGACY_OP=False
export RL_LOGGING_LEVEL=INFO
export TOKENIZERS_PARALLELISM=true
export MAX_IDLE_COUNT=1000
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export BATCHED_PIPE_TIMEOUT=60

# ==================================================================================== #
# ================================ Install dependencies ============================== #

printf "* Installing dependencies\n"

# Install base dependencies
uv_pip_install \
  hypothesis \
  future \
  cloudpickle \
  packaging \
  pygame \
  "moviepy<2.0.0" \
  tqdm \
  pytest \
  pytest-cov \
  pytest-mock \
  pytest-instafail \
  pytest-rerunfailures \
  pytest-timeout \
  pytest-forked \
  pytest-asyncio \
  expecttest \
  "pybind11[global]>=2.13" \
  pyyaml \
  scipy \
  hydra-core \
  tensorboard \
  "imageio==2.26.0" \
  "huggingface-hub>=0.34.0,<1.0" \
  wandb \
  mlflow \
  av \
  coverage \
  transformers \
  ninja \
  timm

# Install dm_control for Python < 3.13
# labmaze (dm_control dependency) doesn't have Python 3.13+ wheels
if [[ "$PYTHON_VERSION" != "3.13" && "$PYTHON_VERSION" != "3.14" ]]; then
  echo "installing dm_control"
  uv_pip_install dm_control
fi

# Install ray for Python < 3.14 (ray doesn't support Python 3.14 yet)
if [[ "$PYTHON_VERSION" != "3.14" ]]; then
  echo "installing ray"
  uv_pip_install ray
fi

# Install mujoco for Python < 3.14 (mujoco doesn't have Python 3.14 wheels yet)
if [[ "$PYTHON_VERSION" != "3.14" ]]; then
  echo "installing mujoco"
  uv_pip_install "mujoco>=3.3.7"
fi

# Install gymnasium
echo "installing gymnasium"
if [[ "$PYTHON_VERSION" == "3.14" ]]; then
  # Python 3.14: no mujoco wheels available, ale_py also failing
  uv_pip_install "gymnasium>=1.1"
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
  uv_pip_install ale-py sympy
  uv_pip_install "gymnasium[mujoco]>=1.1" "mo-gymnasium[mujoco]"
else
  uv_pip_install "gymnasium[atari,mujoco]>=1.1" "mo-gymnasium[mujoco]"
fi

# sanity check
if [[ "$PYTHON_VERSION" != "3.13" && "$PYTHON_VERSION" != "3.14" ]]; then
  python -c "
import dm_control
from dm_control import composer
from tensorboard import *
from google.protobuf import descriptor as _descriptor
"
else
  python -c "
from tensorboard import *
from google.protobuf import descriptor as _descriptor
"
fi

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #

unset PYTORCH_VERSION

if [ "${CU_VERSION:-}" == cpu ] ; then
    echo "Using cpu build"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
fi

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      uv_pip_install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
  else
      uv_pip_install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      uv_pip_install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
      uv_pip_install torch torchvision --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# smoke test
python -c "import functorch"

# Help CMake find pybind11 when building tensordict from source.
# pybind11 ships a CMake package; its location can be obtained via `python -m pybind11 --cmakedir`.
pybind11_DIR="$(python -m pybind11 --cmakedir)"
export pybind11_DIR

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv_pip_install --no-build-isolation git+https://github.com/pytorch/tensordict.git
else
  uv_pip_install tensordict
fi

printf "* Installing torchrl\n"
if [[ "$RELEASE" == 0 ]]; then
  uv_pip_install -e . --no-build-isolation --no-deps
else
  uv_pip_install -e . --no-build-isolation
fi

if [ "${CU_VERSION:-}" != cpu ] ; then
  printf "* Installing VC1\n"
  # Install vc_models directly via uv.
  # VC1Transform.install_vc_models() uses `setup.py develop` which expects `pip`
  # to be present in the environment, but uv-created venvs do not necessarily
  # ship with pip.
  uv_pip_install "git+https://github.com/facebookresearch/eai-vc.git#subdirectory=vc_models"

  printf "* Upgrading timm\n"
  # Keep HF Hub constrained: timm can pull a hub>=1.x which breaks transformers'
  # import-time version check.
  uv_pip_install --upgrade "timm>=0.9.0" "huggingface-hub>=0.34.0,<1.0"

  python -c "
import vc_models
from vc_models.models.vit import model_utils
print(model_utils)
"
fi

# ==================================================================================== #
# ================================ Run tests ========================================= #

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env

Xvfb :99 -screen 0 1024x768x24 &

pytest test/smoke_test.py -v --durations 200
pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'

# Track if any tests fail
EXIT_STATUS=0

# Run distributed tests first (GPU only) to surface errors early
if [ "${CU_VERSION:-}" != cpu ] ; then
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_distributed.py \
    --instafail --durations 200 -vv --capture no \
    --timeout=120 --mp_fork_if_no_cuda || EXIT_STATUS=$?
fi

# Run remaining tests (always run even if distributed tests failed)
if [ "${CU_VERSION:-}" != cpu ] ; then
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/test_distributed.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda || EXIT_STATUS=$?
else
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/test_distributed.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda || EXIT_STATUS=$?
fi

# Fail the workflow if any tests failed
if [ $EXIT_STATUS -ne 0 ]; then
  echo "Some tests failed with exit status $EXIT_STATUS"
fi

coverage combine
coverage xml -i

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh

# Exit with failure if any tests failed
exit $EXIT_STATUS
