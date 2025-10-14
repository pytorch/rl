#!/usr/bin/env bash

set -e

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

apt-get update && apt-get remove swig -y && apt-get install -y git gcc patchelf libosmesa6-dev libgl1-mesa-glx libglfw3 swig3.0 cmake
ln -s /usr/bin/swig3.0 /usr/bin/swig

# we install d4rl here bc env variables have been updated
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
#uv pip install -U 'mujoco-py<2.1,>=2.0'
uv pip install -U "gym[classic_control,atari,accept-rom-license]"==0.23
uv pip install -U six
uv pip install -e . --no-build-isolation
cd ..

#flow is a dependency disaster of biblical scale
#git clone https://github.com/flow-project/flow.git
#cd flow
#uv pip install -e . --no-build-isolation
#cd ..

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
printf "* Smoke test\n"

python -c """import gym
import d4rl
"""

printf "* Tests"
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestD4RL --error-for-skips --runslow
coverage combine
coverage xml -i
