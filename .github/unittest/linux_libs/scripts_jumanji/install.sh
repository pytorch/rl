#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

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

printf "Installing PyTorch with cu121"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install torch --index-url https://download.pytorch.org/whl/cpu
  else
      pip3 install torch --index-url https://download.pytorch.org/whl/cu121
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  pip3 install git+https://github.com/pytorch/tensordict.git
else
  pip3 install tensordict
fi

# smoke test
python -c "import functorch;import tensordict"

printf "* Installing torchrl\n"
python setup.py develop

# smoke test
python -c "import torchrl"
