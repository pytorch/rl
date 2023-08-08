#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env
pip install glfw

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
# export PYGLFW_LIBRARY="/usr/lib64/libglfw.so.3"

# echo "TESTING GYM REGKEYS"
# echo "MUJOCO_GL"
# echo $MUJOCO_GL
# # python -c "import glfw"
# python -c "from ale_py import ALEInterface"
python -c "import ale_py; import shimmy; import gymnasium; print(gymnasium.envs.registration.registry.keys())"

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
# python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest --instafail -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py
coverage combine
coverage xml -i
