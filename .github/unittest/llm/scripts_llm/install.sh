#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

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

# We skip pytorch install due to vllm requirements
#printf "Installing PyTorch with cu128"
#if [[ "$TORCH_VERSION" == "nightly" ]]; then
#  if [ "${CU_VERSION:-}" == cpu ] ; then
#      pip install --pre torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/nightly/cpu -U
#  else
#      pip install --pre torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/nightly/cu128 -U
#  fi
#elif [[ "$TORCH_VERSION" == "stable" ]]; then
#    if [ "${CU_VERSION:-}" == cpu ] ; then
#      pip install torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/cpu
#  else
#      pip install torch "numpy<2.0.0" --index-url https://download.pytorch.org/whl/cu128
#  fi
#else
#  printf "Failed to install pytorch"
#  exit 1
#fi

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  pip install "pybind11[global]" ninja
  pip install git+https://github.com/pytorch/tensordict.git
else
  pip install tensordict
fi

# smoke test
python -c "import tensordict"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation

# smoke test
python -c "import torchrl"

# Install MCP dependencies for tool execution tests
printf "* Installing MCP dependencies (uvx, Deno)\n"

# Install uv (provides uvx command)
pip install uv

# Install Deno (required by mcp-run-python)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"

# Verify installations
uvx --version || echo "Warning: uvx not installed"
deno --version || echo "Warning: Deno not installed"

# Pre-download models for LLM tests to avoid timeout during test execution
printf "* Pre-downloading models for LLM tests\n"
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')"

# Pre-warm SGLang by launching a server once to compile CUDA kernels
# This is critical for fast startup during tests
printf "* Pre-warming SGLang server (compiling CUDA kernels)...\n"
python -c "
import subprocess
import time
import requests
import sys

# Start SGLang server in background
print('Starting SGLang server for pre-warming...')
proc = subprocess.Popen(
    ['python3', '-m', 'sglang.launch_server',
     '--model-path', 'Qwen/Qwen2.5-0.5B',
     '--host', '127.0.0.1',
     '--port', '39999',
     '--tp-size', '1',
     '--mem-fraction-static', '0.3'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

# Wait for server to be ready (up to 10 minutes for first-time kernel compilation)
start = time.time()
timeout = 600
ready = False
while time.time() - start < timeout:
    try:
        resp = requests.get('http://127.0.0.1:39999/health', timeout=5)
        if resp.status_code == 200:
            ready = True
            break
    except:
        pass
    time.sleep(2)

elapsed = time.time() - start
if ready:
    print(f'SGLang server ready after {elapsed:.1f}s - kernels compiled and cached')
else:
    print(f'WARNING: SGLang server did not start within {timeout}s')
    # Print server output for debugging
    proc.terminate()
    proc.wait(timeout=5)
    sys.exit(1)

# Shutdown server
proc.terminate()
proc.wait(timeout=10)
print('SGLang pre-warming complete')
"
