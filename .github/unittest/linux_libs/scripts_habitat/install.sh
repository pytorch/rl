#!/usr/bin/env bash

unset PYTORCH_VERSION

set -e
set -v

# Make uv available (installed in setup_env.sh)
export PATH="$HOME/.local/bin:$PATH"

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

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
  uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv pip install git+https://github.com/pytorch/tensordict.git
else
  uv pip install tensordict
fi

# smoke test
python -c "import tensordict"

# Install build dependencies (required for --no-build-isolation)
printf "* Installing build dependencies\n"
uv pip install setuptools wheel ninja "pybind11[global]" cmake

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation

# smoke test
 -c "import torchrl"
