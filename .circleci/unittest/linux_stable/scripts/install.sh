#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
    cudatoolkit="cpuonly"
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
    cudatoolkit="cudatoolkit=${version}"
fi

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [ "${CU_VERSION:-}" == cpu ] ; then
    # conda install -y pytorch torchvision cpuonly -c pytorch-nightly
    # use pip to install pytorch as conda can frequently pick older release
#    conda install -y pytorch torchvision cpuonly -c pytorch
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
else
#    conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
fi

printf "Installing functorch\n"
pip3 install functorch

# smoke test
python -c "import functorch"

printf "* Installing torchrl\n"
printf "g++ version: "
gcc --version

python setup.py develop
