#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

root_dir="$(git rev-parse --show-toplevel)"
# Add uv to PATH (it was installed in setup_env.sh)
export PATH="$HOME/.local/bin:$PATH"
source "${root_dir}/.venv/bin/activate"

# Install build dependencies EARLY (required for --no-build-isolation)
printf "* Installing build dependencies\n"
uv pip install setuptools wheel ninja "pybind11[global]"

if [ "${CU_VERSION:-}" == cpu ] ; then
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
fi

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with cu128"
if [ "${CU_VERSION:-}" == cpu ] ; then
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu -U
else
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U
fi

# smoke test
python -c "import functorch"

# install tensordict
uv pip install git+https://github.com/pytorch/tensordict

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation
