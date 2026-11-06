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
    python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
else
    python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "https://download.pytorch.org/whl/${CU_VERSION}"
fi

# Solving circular import: https://stackoverflow.com/questions/75501048/how-to-fix-attributeerror-partially-initialized-module-charset-normalizer-has
#pip install -U charset-normalizer

# install tensordict (from source on PR/nightly, from PyPI only on release/*)
if [[ "$RELEASE" == 0 ]]; then
  # pybind11 headers are required to build tensordict's C++ extension.
  python -m pip install "pybind11[global]"
  python -m pip install cloudpickle packaging importlib_metadata numpy 'orjson; python_version < "3.13"' "pyvers>=0.2.3,<0.3.0"
  python -m pip install --no-deps git+https://github.com/pytorch/tensordict.git
else
  python -m pip install cloudpickle packaging importlib_metadata numpy 'orjson; python_version < "3.13"' "pyvers>=0.2.3,<0.3.0"
  python -m pip install --no-deps "tensordict>=0.15.2,<0.16.0"
fi

# smoke test
python -c "import tensordict; print(f'tensordict: {tensordict.__version__}')"

printf "* Installing hoptorch\n"
python -m pip install "hoptorch>=0.1.4"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation --no-deps
python -c "import torchrl"
