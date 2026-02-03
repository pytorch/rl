#!/usr/bin/env bash

# Install script for SGLang tests.
# This uses uv and installs SGLang WITHOUT vLLM to avoid Triton version conflicts.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"

# Activate environment
source "${env_dir}/bin/activate"

# submodules
git submodule sync && git submodule update --init --recursive

# ============================================================================================ #
# ================================ Install dependencies ====================================== #
# ============================================================================================ #

printf "* Installing base dependencies\n"
uv pip install \
    hypothesis \
    future \
    cloudpickle \
    pytest \
    pytest-cov \
    pytest-mock \
    pytest-instafail \
    pytest-rerunfailures \
    pytest-json-report \
    pytest-asyncio \
    pytest-timeout \
    expecttest \
    pyyaml \
    scipy \
    hydra-core

# ============================================================================================ #
# ================================ PyTorch Installation ====================================== #
# ============================================================================================ #

printf "* Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
    if [ "${CU_VERSION:-}" == cpu ]; then
        uv pip install --upgrade --pre torch torchvision "numpy<2.0.0" --index-url https://download.pytorch.org/whl/nightly/cpu
    else
        uv pip install --upgrade --pre torch torchvision "numpy<2.0.0" --index-url https://download.pytorch.org/whl/nightly/${CU_VERSION}
    fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ]; then
        uv pip install --upgrade torch torchvision "numpy<2.0.0" --index-url https://download.pytorch.org/whl/cpu
    else
        uv pip install --upgrade torch torchvision "numpy<2.0.0" --index-url https://download.pytorch.org/whl/${CU_VERSION}
    fi
else
    printf "Failed to install pytorch\n"
    exit 1
fi

# ============================================================================================ #
# ================================ TensorDict Installation =================================== #
# ============================================================================================ #

printf "* Installing tensordict\n"
uv pip install "pybind11[global]" ninja
if [[ "$RELEASE" == 0 ]]; then
    uv pip install --no-deps git+https://github.com/pytorch/tensordict.git
else
    uv pip install --no-deps tensordict
fi

# smoke test
python -c "import tensordict"

# ============================================================================================ #
# ================================ TorchRL Installation ====================================== #
# ============================================================================================ #

printf "* Installing torchrl\n"
uv pip install -e . --no-build-isolation --no-deps

# smoke test
python -c "import torchrl"

# ============================================================================================ #
# ================================ SGLang Installation ======================================= #
# ============================================================================================ #

printf "* Installing SGLang dependencies\n"
uv pip install transformers accelerate datasets

# Install SGLang with all extras
# Note: We do NOT install vLLM here to avoid Triton version conflicts
printf "* Installing SGLang\n"
uv pip install "sglang[all]"

# Install MCP dependencies for tool execution tests
printf "* Installing MCP dependencies (uvx, Deno)\n"

# Install Deno (required by mcp-run-python)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"

# Install mcp
uv pip install mcp langdetect

# Verify installations
deno --version || echo "Warning: Deno not installed"

# Pre-download models for LLM tests to avoid timeout during test execution
printf "* Pre-downloading models for LLM tests\n"
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')"

printf "* SGLang installation complete\n"

# Show installed versions for debugging
printf "* Installed versions:\n"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sglang; print(f'SGLang: {sglang.__version__}')" || echo "SGLang version check failed"
python -c "import triton; print(f'Triton: {triton.__version__}')" || echo "Triton version check failed"
