#!/usr/bin/env bash

set -euxo pipefail
set -v
set -e

# =============================================================================== #
# ================================ Init ========================================= #


if [[ $OSTYPE != 'darwin'* ]]; then
  apt-get update && apt-get upgrade -y
  apt-get install -y vim git wget curl cmake

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

# 3. Install dependencies for OPTDEPS testing
# Optdeps tests that torchrl works WITHOUT optional dependencies
# We install: build deps + test deps only (NO optional deps like gym, transformers, etc.)
printf "* Installing build and test dependencies (NO optional deps)\n"

# Install build dependencies FIRST (required for C++ extensions)
printf "* Installing build dependencies\n"
uv pip install setuptools wheel ninja "pybind11[global]"

# Install test dependencies from requirements.txt (pytest, hypothesis, etc.)
printf "* Installing test dependencies from requirements.txt\n"
uv pip install -r "${this_dir}/requirements.txt"

# Install pip for compatibility with packages that expect it
uv pip install pip

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
      uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install torch --index-url https://download.pytorch.org/whl/cpu -U
  else
      uv pip install torch --index-url https://download.pytorch.org/whl/$CU_VERSION -U
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

printf "* Installing torchrl (will install CORE dependencies only)\n"
# Standard installation: installs [project.dependencies] but NOT [project.optional-dependencies]
# Core deps: torch, numpy, packaging, cloudpickle, tensordict (all pre-installed above)
# --no-build-isolation: build deps already available, avoids reinstalling torch in temp env
uv pip install -e . --no-build-isolation

# smoke test
python -c "import torchrl"

# ==================================================================================== #
# ================================ Run tests ========================================= #


# find libstdc (if needed)
STDC_LOC=$(find ${env_dir}/ -name "libstdc++.so.6" 2>/dev/null | head -1 || echo "")

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"

export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=100
export BATCHED_PIPE_TIMEOUT=60

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
  --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
  --ignore test/test_distributed.py \
  --ignore test/llm \
  --timeout=120 --mp_fork_if_no_cuda

coverage combine
coverage xml -i

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
