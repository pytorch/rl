#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.
apt-get update && apt-get install -y \
    cmake \
    git \
    wget \
    gcc \
    g++ \
    patchelf \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    swig3.0 \
    libglew-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2

set -e

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

if [ "${CU_VERSION:-}" == cpu ] ; then
    version="cpu"
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

printf "Installing PyTorch with cu128"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
      uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv pip install git+https://github.com/pytorch/tensordict.git
else
  uv pip install tensordict
fi

# smoke test
python -c "import functorch;import tensordict"

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation

# smoke test
python -c "import torchrl"
