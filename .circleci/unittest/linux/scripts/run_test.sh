#!/usr/bin/env bash

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python3 -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch

# install vc1
python3 -c """
from torchrl.envs.transforms.vc1 import VC1Transform
VC1Transform.install_vc_models(auto_exit=True)
"""

python3 -c """
import vc_models
from vc_models.models.vit import model_utils
print(model_utils)
"""

python3 .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
python3 .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'
python3 .circleci/unittest/helpers/coverage_run_parallel.py -m pytest --instafail -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py -k VC1
python3 .circleci/unittest/helpers/coverage_run_parallel.py -m pytest --instafail -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py33
coverage combine
coverage xml -i
