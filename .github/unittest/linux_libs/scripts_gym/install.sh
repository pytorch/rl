#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.
apt-get update && apt-get install -y git wget gcc g++
set -e
set -v

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

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
    uv pip install torch==2.0 torchvision==0.15 --index-url https://download.pytorch.org/whl/cpu
else
    uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 numpy==1.26 --index-url https://download.pytorch.org/whl/cu118
fi

# Solving circular import: https://stackoverflow.com/questions/75501048/how-to-fix-attributeerror-partially-initialized-module-charset-normalizer-has
uv pip install -U charset-normalizer

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv pip install "cmake>=3.22"
  uv pip install "pybind11[global]"
  uv pip install git+https://github.com/pytorch/tensordict.git
else
  uv pip install tensordict
fi

# smoke test
python -c "import tensordict"

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation
python -c "import torchrl"

## Reinstalling pytorch with specific version
#printf "Re-installing PyTorch with %s\n" "${CU_VERSION}"
#if [ "${CU_VERSION:-}" == cpu ] ; then
#    conda install pytorch==1.13.1 torchvision==0.14.1 cpuonly -c pytorch
#else
#    conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
#fi
