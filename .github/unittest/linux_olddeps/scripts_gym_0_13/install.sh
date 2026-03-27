#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

#apt-get update -y && apt-get install git wget gcc g++ -y

if [ "${CU_VERSION:-}" == cpu ] ; then
    cudatoolkit="cpuonly"
    version="cpu"
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
    conda install pytorch==2.1 torchvision==0.16 cpuonly -c pytorch -y
else
    python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
#    conda install pytorch==2.1 torchvision==0.16 pytorch-cuda=11.8 "numpy<2.0" -c pytorch -c nvidia -y
fi

# Solving circular import: https://stackoverflow.com/questions/75501048/how-to-fix-attributeerror-partially-initialized-module-charset-normalizer-has
#pip install -U charset-normalizer

# install tensordict
#
# NOTE:
# - The olddeps CI job runs on older Python/torch stacks.
# - Installing from tensordict `main` (git+https) is brittle as `main` may drop
#   support for older Python versions at any time, which can lead to "tensordict
#   not installed" failures in downstream smoke tests.
# - Use the same (pinned) range as TorchRL itself to keep this job stable.
python -m pip install "${TORCHRL_TENSORDICT_SPEC:-tensordict>=0.11.0,<0.12.0}"

# smoke test
python -c "import tensordict; print(f'tensordict: {tensordict.__version__}')"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation
python -c "import torchrl"
