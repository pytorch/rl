#!/usr/bin/env bash

unset PYTORCH_VERSION

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
    CUDA_VERSION="cpu"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
fi

git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch\n"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install torch --index-url https://download.pytorch.org/whl/cpu -U
  else
      pip3 install torch --index-url https://download.pytorch.org/whl/cu128
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# Install mjlab after PyTorch so its torch>=2.7 dependency is satisfied by the
# CI-selected wheel rather than pulling a second torch build from PyPI.
pip install "mjlab>=1.4.0" --progress-bar off
pip install git+https://github.com/pytorch/tensordict.git --progress-bar off

python -c "import torch; import tensordict; import mjlab"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation
python -c "import torchrl"
