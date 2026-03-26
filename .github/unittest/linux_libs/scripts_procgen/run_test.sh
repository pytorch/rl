#!/usr/bin/env bash

set -e

# Activate the virtual environment
source ./env/bin/activate

apt-get update && apt-get install -y git wget cmake

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

deactivate 2>/dev/null || true && source ./env/bin/activate

# this workflow only tests the libs
python -c "import procgen"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestProcgen --error-for-skips --runslow

coverage combine -q
coverage xml -i
