#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -ex

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

conda activate torchrl

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
git submodule sync && git submodule update --init --recursive
python -m pip install "numpy<2.0"

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if $torch_cuda ; then
    python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
  else
    python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  if $torch_cuda ; then
      python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
  else
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
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
if [[ "$RELEASE" == 0 ]]; then
  git clone https://github.com/pytorch/tensordict
  cd tensordict
  python setup.py develop
  cd ..
else
  pip3 install tensordict
fi

# smoke test
python -c """
from tensordict import TensorDict
print('successfully imported tensordict')
"""

source "$this_dir/set_cuda_envs.sh"

printf "* Installing torchrl\n"
python setup.py develop

whatsinside=$(ls -rtlh ./torchrl)
echo $whatsinside

# smoke test
python -c """
from torchrl.data import ReplayBuffer
print('successfully imported torchrl')
"""
