#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env


export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False

# JAX (mjx backend) GPU initialization: mirrors scripts_brax/run_test.sh.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# OpenGL backend for mujoco.Renderer (used by the mjx and mujoco backends'
# from_pixels path; the mujoco-torch backend uses its own torch raycaster
# and doesn't need a GL context). EGL works headless on the GPU runner;
# matches scripts_gym/run_all.sh.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m torch.utils.collect_env
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

export MKL_THREADING_LAYER=GNU
export MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON

# Lib smoke checks
python -c "import mujoco; import mujoco.mjx; import mujoco_torch; print('mujoco', mujoco.__version__)"
python -c "
import jax
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
try:
    devices = jax.devices()
    print(f'JAX devices: {devices}')
except Exception as e:
    print(f'JAX init error: {e}; falling back to CPU')
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    jax.config.update('jax_platform_name', 'cpu')
"
python -c 'import torch;t = torch.ones([2,2], device="cuda:0" if torch.cuda.is_available() else "cpu");print(t);print("tensor device:" + str(t.device))'

# JSON report for flaky test tracking
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-mujoco.json --json-report-indent=2"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs/test_mujoco.py ${json_report_args} --instafail -v --durations 200 --capture no -k TestMujoco --error-for-skips
coverage combine -q
coverage xml -i

# Upload test results with metadata for flaky tracking
python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
