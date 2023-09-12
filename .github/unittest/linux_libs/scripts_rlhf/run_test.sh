#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt-get update && apt-get install -y git gcc
ln -s /usr/bin/swig3.0 /usr/bin/swig

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

conda deactivate && conda activate ./env

python -c "import transformers, datasets"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_rlhf.py --instafail -v --durations 200 --capture no --error-for-skips
coverage combine
coverage xml -i
