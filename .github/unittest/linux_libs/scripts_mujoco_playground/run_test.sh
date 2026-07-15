#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env


export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False

# Configure JAX for proper GPU initialization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

export MKL_THREADING_LAYER=GNU
export MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON

# this workflow only tests the libs
python -c "import mujoco_playground"
python -c "from mujoco_playground import dm_control_suite, locomotion, manipulation, registry"

# Report JAX devices. We deliberately avoid try/except + JAX_PLATFORM_NAME
# fallback here: it runs in a subprocess that exits immediately, so it has no
# effect on the pytest invocation below, and it would hide a real "GPU not
# visible" failure from CI.
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python -c 'import torch;t = torch.ones([2,2], device="cuda:0");print(t);print("tensor device:" + str(t.device))'

# JSON report for flaky test tracking
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-mujoco_playground.json --json-report-indent=2"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs ${json_report_args} --instafail -v --durations 200 --capture no -k TestMujocoPlayground --error-for-skips
coverage combine -q
coverage xml -i

# Upload test results with metadata for flaky tracking
python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
