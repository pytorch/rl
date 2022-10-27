#!/usr/bin/env bash

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

pytest test/smoke_test.py -v --durations 20
pytest test/smoke_test_deps.py -v --durations 20 -k 'test_gym or test_dm_control_pixels or test_dm_control'
#xvfb-run -s ":99 -screen 0 1280x1024x24" pytest --instafail -v --durations 20
#pytest --instafail -v --durations 20
python test/test_libs.py
