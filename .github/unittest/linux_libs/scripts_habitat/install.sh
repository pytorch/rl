#!/usr/bin/env bash

unset PYTORCH_VERSION

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [[ ${#CU_VERSION} -eq 4 ]]; then
    CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
elif [[ ${#CU_VERSION} -eq 5 ]]; then
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
fi
echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 -U
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  pip3 install torch --index-url https://download.pytorch.org/whl/cu121
fi

# install tensordict
# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  pip3 install git+https://github.com/pytorch/tensordict.git
else
  pip3 install tensordict
fi

# smoke test
python3 -c "import functorch;import tensordict"

printf "* Installing torchrl\n"
python setup.py develop

# smoke test
python3 -c "import torchrl"
