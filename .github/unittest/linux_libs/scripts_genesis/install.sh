#!/usr/bin/env bash

unset PYTORCH_VERSION

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

git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with cu128"
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
      pip3 install torch --index-url https://download.pytorch.org/whl/cu128 -U
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

if [[ "$RELEASE" == 0 ]]; then
  pip3 install git+https://github.com/pytorch/tensordict.git
else
  pip3 install tensordict
fi

python -c "import tensordict"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation

python -c "import torchrl"

printf "* Installing genesis-world\n"
pip3 install genesis-world

python -c "import genesis; print('Genesis installed successfully')"
