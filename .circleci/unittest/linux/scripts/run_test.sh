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

coverage run -m pytest test/smoke_test.py -v --durations 20
coverage run -m pytest test/smoke_test_deps.py -v --durations 20 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'
coverage run -m pytest --instafail -v --durations 20
coverage xml -i
