#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt-get update && apt-get install -y git wget

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

conda deactivate && conda activate ./env

# this workflow only tests the libs
python -c "import chess"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_env.py --instafail -v --durations 200 --capture no -k TestChessEnv --error-for-skips --runslow

coverage combine
coverage xml -i
