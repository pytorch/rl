#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

# Start Xvfb with specific OpenGL configuration
export DISPLAY=:99
Xvfb :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset > /dev/null 2>&1 &
sleep 3  # Give Xvfb time to start

# Verify OpenGL/EGL setup
glxinfo -B || true
echo "EGL_PLATFORM=$EGL_PLATFORM"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"

# Run the tests
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym'

unset LD_PRELOAD
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 -k "gym and not isaac" --error-for-skips --mp_fork
coverage combine
coverage xml -i
