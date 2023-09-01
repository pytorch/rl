#!/usr/bin/env bash

# this code is supposed to run on CPU
# rendering with the combination of packages we have here in headless mode
# is hard to nail.
# IMPORTANT: As a consequence, we can't guarantee TorchRL compatibility with
# rendering with this version of gym / mujoco-py.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

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

# this workflow only tests the libs
python -c "import envpool"

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestEnvPool --error-for-skips
coverage combine
coverage xml -i
