#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env
apt-get update && apt-get install -y git wget


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
# more logging
export MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON

#wget https://github.com/openai/mujoco-py/blob/master/vendor/10_nvidia.json
#mv 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# this workflow only tests the libs
python -c "import brax"

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestBrax --error-for-skips
coverage combine
coverage xml -i
