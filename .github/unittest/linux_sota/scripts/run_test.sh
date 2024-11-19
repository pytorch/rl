#!/usr/bin/env bash

set -e
set -v

# Initialize an error flag
error_occurred=0
# Function to handle errors
error_handler() {
    echo "Error on line $1"
    error_occurred=1
}
# Trap ERR to call the error_handler function with the failing line number
trap 'error_handler $LINENO' ERR

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
export CUDA_LAUNCH_BLOCKING=1

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200

coverage run -m pytest .github/unittest/linux_sota/scripts/test_sota.py --instafail --durations 200 -vvv --capture no

coverage combine
coverage xml -i
