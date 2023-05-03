#!/usr/bin/env bash

set -e
set -v

# Extracted from run_test.sh to run once.
apt-get update && apt-get install -y git wget libglew-dev libx11-dev x11proto-dev g++

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 20
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 20 -k 'test_gym'

export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 20 -k "robohive" --error-for-skips
coverage combine
coverage xml -i
