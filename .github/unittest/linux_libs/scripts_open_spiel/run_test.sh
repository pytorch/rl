#!/usr/bin/env bash

set -e

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

apt-get update && apt-get install -y git wget cmake

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# conda deactivate (not needed with uv) && source ./.venv/bin/activate

# this workflow only tests the libs
python -c "import pyspiel"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestOpenSpiel --error-for-skips --runslow

coverage combine
coverage xml -i
