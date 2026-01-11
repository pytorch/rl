#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -ex
# =================================== Setup =================================================

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

cd "${root_dir}"

echo "=== Starting Windows CI setup ==="
echo "Current directory: $(pwd)"
echo "Python version: $PYTHON_VERSION"
echo "CU_VERSION: $CU_VERSION"
echo "TORCH_VERSION: $TORCH_VERSION"

eval "$($(which conda) shell.bash hook)" && set -x

# Create test environment at ./env
printf "* Creating a test environment\n"
conda create --name ci -y python="$PYTHON_VERSION"

printf "* Activating the environment"
conda activate ci

printf "Python version"
echo $(which python)
echo $(python --version)
echo $(conda info -e)

echo "=== Installing test dependencies ==="
python -m pip install hypothesis future cloudpickle pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures expecttest pyyaml scipy coverage

# =================================== Install =================================================

echo "=== Installing PyTorch and dependencies ==="

# TODO, refactor the below logic to make it easy to understand how to get correct cuda_version.
if [ "${CU_VERSION:-}" == cpu ] ; then
    cudatoolkit="cpuonly"
    version="cpu"
    torch_cuda="False"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi

    cuda_toolkit_pckg="cudatoolkit"
    if [[ $CUDA_VERSION == 11.6 || $CUDA_VERSION == 11.7 ]]; then
        cuda_toolkit_pckg="pytorch-cuda"
    fi

    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="${cuda_toolkit_pckg}=${version}"
fi

# submodules
echo "=== Updating git submodules ==="
git submodule sync && git submodule update --init --recursive
python -m pip install "numpy<2.0"

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if $torch_cuda ; then
    python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118 -U
  else
    python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  if $torch_cuda ; then
      python -m pip install torch --index-url https://download.pytorch.org/whl/cu118 -U
  else
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu -U
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

torch_cuda=$(python -c "import torch; print(torch.cuda.is_available())")
echo torch.cuda.is_available is $torch_cuda

if [ ! -z "${CUDA_VERSION:-}" ] ; then
    if [ "$torch_cuda" == "False" ]; then
        echo "torch with cuda installed but torch.cuda.is_available() is False"
        exit 1
    fi
fi

#python -m pip install pip --upgrade

# install tensordict
echo "=== Installing tensordict ==="
if [[ "$RELEASE" == 0 ]]; then
  conda install anaconda::cmake -y

  python -m pip install "pybind11[global]"

  python -m pip install git+https://github.com/pytorch/tensordict
else
  pip3 install tensordict
fi

# smoke test
echo "=== Testing tensordict import ==="
python -c """
from tensordict import TensorDict
print('successfully imported tensordict')
"""

echo "=== Setting up CUDA environment ==="
source "$this_dir/set_cuda_envs.sh"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation

whatsinside=$(ls -rtlh ./torchrl)
echo $whatsinside

# smoke test
echo "=== Testing torchrl import ==="
python -c """
from torchrl.data import ReplayBuffer
print('successfully imported torchrl')
"""

# =================================== Run =================================================

echo "=== Setting up test environment ==="
source "$this_dir/set_cuda_envs.sh"

# we don't use torchsnapshot
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=60
export BATCHED_PIPE_TIMEOUT=60
export LAZY_LEGACY_OP=False

echo "=== Collecting environment info ==="
python -m torch.utils.collect_env

echo "=== Starting pytest execution ==="
echo "Current working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Pytest executable: $(which pytest)"

# Create test-results directory if it doesn't exist
mkdir -p test-results

# Run pytest with explicit error handling
set +e  # Don't exit on error for pytest
pytest --junitxml=test-results/junit.xml -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py --ignore test/llm
PYTEST_EXIT_CODE=$?
set -e  # Re-enable exit on error

echo "=== Pytest completed with exit code: $PYTEST_EXIT_CODE ==="

# Exit with pytest's exit code
exit $PYTEST_EXIT_CODE
