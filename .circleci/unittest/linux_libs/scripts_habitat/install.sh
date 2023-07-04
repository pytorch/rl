#!/usr/bin/env bash

unset PYTORCH_VERSION

set -e
set -v

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
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall
else
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu116 --force-reinstall
fi

# install tensordict
#mkdir "third_party"
#cd third_party
#git clone https://github.com/pytorch-labs/tensordict
#cd tensordict
#python setup.py develop
#cd ../..
pip3 install tensordict-nightly

# smoke test
python3 -c "import functorch;import tensordict"

printf "* Installing torchrl\n"
pip3 install -e .

# smoke test
python3 -c "import torchrl"
