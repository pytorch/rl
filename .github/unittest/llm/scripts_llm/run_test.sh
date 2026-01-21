#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt-get update && apt-get install -y git gcc cmake
ln -s /usr/bin/swig3.0 /usr/bin/swig

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False

# to solve RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

conda deactivate && conda activate ./env

# Run pytest with:
# - --runslow: Run slow tests that would otherwise skip
# - --ignore: Exclude tests requiring unavailable dependencies (mlgym not on PyPI)
# - --timeout: 5 minute timeout per test to prevent hangs
# Note: Removed --error-for-skips as many LLM tests use pytest.skip for optional dependencies
# Note: Removed --exitfirst to run all tests and collect all failures
pytest test/llm -vvv --instafail --durations 600 --capture no --timeout=300 \
    --runslow \
    --ignore=test/llm/libs/test_mlgym.py
