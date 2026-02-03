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

# JSON report for flaky test tracking
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-llm.json --json-report-indent=2"

# Run pytest with:
# - --runslow: Run slow tests that would otherwise skip
# - --ignore: Exclude tests requiring unavailable dependencies (mlgym not on PyPI)
# - --ignore: Exclude SGLang tests (run in separate workflow due to Triton conflicts)
# - --timeout: 5 minute timeout per test to prevent hangs
# Note: Removed --isolate (too slow - each test in subprocess adds huge overhead)
# Note: Removed --error-for-skips as many LLM tests use pytest.skip for optional dependencies
# Note: Removed --exitfirst to run all tests and collect all failures
pytest test/llm ${json_report_args} -vvv --instafail --durations 600 --capture no --timeout=300 \
    --runslow \
    --ignore=test/llm/libs/test_mlgym.py \
    --ignore=test/llm/test_sglang.py \
    --ignore=test/llm/test_sglang_updaters.py

# Upload test results with metadata for flaky tracking
python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
