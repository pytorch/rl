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
        uv pip install --upgrade --pre torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/nightly/cpu
    else
        uv pip install --upgrade --pre torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/nightly/${CU_VERSION}
    fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ]; then
        uv pip install --upgrade torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/cpu
    else
        uv pip install --upgrade torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/${CU_VERSION}
    fi
else
    printf "Failed to install pytorch\n"
    exit 1
fi

# ============================================================================================ #
# ================================ TensorDict Installation =================================== #
# ============================================================================================ #

printf "* Installing tensordict\n"
# Install tensordict dependencies first (pyvers is required but --no-deps skips it)
uv pip install cloudpickle packaging importlib_metadata numpy orjson "pyvers>=0.2.0,<0.3.0"
uv pip install "pybind11[global]" ninja
if [[ "$RELEASE" == 0 ]]; then
    uv pip install --no-build-isolation --no-deps git+https://github.com/pytorch/tensordict.git
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

# Install system dependencies required by SGLang
# libnuma is required by sglang-kernel
printf "* Installing system dependencies for SGLang\n"
apt-get update && apt-get install -y libnuma-dev

# Install SGLang with all extras
# Note: We do NOT install vLLM here to avoid Triton version conflicts
printf "* Installing SGLang\n"
uv pip install "sglang[all]" "kernels>=0.12,<0.13"

# SGLang pins torch 2.11.0, whose PyPI wheel carries CUDA 13.0. Install
# the exact PyPI torchvision build after SGLang resolves torch so the
# initial PyTorch cu129 wheel cannot satisfy the version constraint.
printf "* Installing torchvision matching SGLang's torch wheel\n"
uv pip install --reinstall --index-url https://pypi.org/simple "torchvision===0.26.0"

# Keep secondary dependencies inside the ranges required by the latest SGLang
# dependency set so uv pip check catches real breakage instead of resolver drift.
printf "* Constraining secondary dependencies for SGLang\n"
uv pip install "pillow>=9.2,<12" "numpy>=1.25,<2.4" "fsspec[http]<=2026.2.0"

# Install MCP dependencies for tool execution tests
printf "* Installing MCP dependencies (uvx, Deno)\n"

# Install Deno (required by mcp-run-python)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"

# Install mcp
uv pip install mcp langdetect

# SGLang may resolve a backend-specific torch/triton stack. Reinstall
# TensorDict and TorchRL after that resolution so native extensions are built
# against the final torch wheel present in the environment.
printf "* Reinstalling TensorDict and TorchRL against final backend stack\n"
if [[ "$RELEASE" == 0 ]]; then
    uv pip install --reinstall --no-build-isolation --no-deps git+https://github.com/pytorch/tensordict.git
else
    uv pip install --reinstall --no-deps tensordict
fi
uv pip install --reinstall -e . --no-build-isolation --no-deps

# Verify installations
deno --version || echo "Warning: Deno not installed"

# Pre-download models for LLM tests to avoid timeout during test execution
printf "* Pre-downloading models for LLM tests\n"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B')"

printf "* SGLang installation complete\n"

# Show installed versions for debugging
printf "* Installed versions:\n"
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("sglang", "transformers", "kernels", "torch", "torchvision", "triton", "numpy", "pillow", "fsspec"):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: not installed")
PY

printf "* Verifying torch/torchvision CUDA compatibility\n"
python - <<'PY'
import torch
import torchvision

print(f"torch CUDA: {torch.version.cuda}; torchvision: {torchvision.__version__}")
PY

uv pip check
