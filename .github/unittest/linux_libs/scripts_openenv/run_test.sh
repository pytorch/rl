#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt-get update && apt-get install -y git wget cmake

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
git config --global --add safe.directory '*'

conda deactivate && conda activate ./env

python -c "import openenv"
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs --instafail -v --durations 200 --capture no -k TestOpenEnv --error-for-skips

coverage combine -q
coverage xml -i
