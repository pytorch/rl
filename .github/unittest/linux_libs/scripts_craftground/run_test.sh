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
python -c "import craftground"
java -version

# The Minecraft client needs an X display; use a virtual framebuffer with
# Mesa software rendering.
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs/test_craftground.py --instafail -v --durations 200 --capture no -k TestCraftGround --error-for-skips --runslow

coverage combine -q
coverage xml -i
