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
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
#if [ "${os}" == "MacOSX" ]; then
#    conda install -y -c "pytorch-${UPLOAD_CHANNEL}" "pytorch-${UPLOAD_CHANNEL}"::pytorch "${cudatoolkit}" pytest
#else
#    conda install -y -c "pytorch-${UPLOAD_CHANNEL}" "pytorch-${UPLOAD_CHANNEL}"::pytorch[build="*${version}*"] "${cudatoolkit}" pytest
#fi

if [ "${CU_VERSION:-}" == cpu ] ; then
    # conda install -y pytorch torchvision cpuonly -c pytorch-nightly
    # use pip to install pytorch as conda can frequently pick older release
    pip install torch torchvision # -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --pre
else
    conda install -y pytorch torchvision torchaudio torchtext cudatoolkit=10.2 -c pytorch-nightly
fi

#printf "Installing functorch\n"
#cd third_party/functorch
#WHEELS_FOLDER=${HOME}/project/wheels
#mkdir -p $WHEELS_FOLDER
#PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" python setup.py develop bdist_wheel -d $WHEELS_FOLDER
#cd ../..


printf "* Installing torchrl\n"
python setup.py develop
