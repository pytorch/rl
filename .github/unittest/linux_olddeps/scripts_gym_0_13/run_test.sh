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
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pytorch/rl/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin
export MKL_THREADING_LAYER=GNU
export BATCHED_PIPE_TIMEOUT=60

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control'

export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
CKPT_BACKEND=torch MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest --instafail -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py
#pytest --instafail -v --durations 200
#python test/test_libs.py
coverage combine
coverage xml -i
