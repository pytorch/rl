#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.
apt-get update && apt-get install -y git wget gcc g++
#apt-get update && apt-get install -y git wget freeglut3 freeglut3-dev

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

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [ "${CU_VERSION:-}" == cpu ] ; then
    # conda install -y pytorch torchvision cpuonly -c pytorch-nightly
    # use pip to install pytorch as conda can frequently pick older release
#    conda install -y pytorch cpuonly -c pytorch-nightly
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
else
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu118
fi

# install tensordict
pip install git+https://github.com/pytorch-labs/tensordict.git

# smoke test
python -c "import functorch"

printf "* Installing torchrl\n"
python setup.py develop

# smoke test
python -c "import torchrl"
