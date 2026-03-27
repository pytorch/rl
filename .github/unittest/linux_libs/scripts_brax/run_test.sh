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

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
# more logging
export MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON

#wget https://github.com/openai/mujoco-py/blob/master/vendor/10_nvidia.json
#mv 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# this workflow only tests the libs
python -c "import brax"
python -c "import brax.envs"

# Initialize JAX with proper GPU configuration
python -c "
import jax
import jax.numpy as jnp
import os

# Configure JAX for GPU
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Test JAX GPU availability
try:
    devices = jax.devices()
    print(f'JAX devices: {devices}')
    if len(devices) > 1:
        print('JAX GPU is available')
    else:
        print('JAX CPU only')
except Exception as e:
    print(f'JAX initialization error: {e}')
    # Fallback to CPU
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    jax.config.update('jax_platform_name', 'cpu')
    print('Falling back to JAX CPU')
"

python -c 'import torch;t = torch.ones([2,2], device="cuda:0");print(t);print("tensor device:" + str(t.device))'

# JSON report for flaky test tracking
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-brax.json --json-report-indent=2"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py ${json_report_args} --instafail -v --durations 200 --capture no -k TestBrax --error-for-skips
coverage combine -q
coverage xml -i

# Upload test results with metadata for flaky tracking
python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
