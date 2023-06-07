#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt-get update && apt-get remove swig -y && apt-get install -y git gcc patchelf libosmesa6-dev libgl1-mesa-glx libglfw3 swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig

# we install d4rl here bc env variables have been updated
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip3 install -U 'mujoco-py<2.1,>=2.0'
pip3 install -U "gym[classic_control,atari,accept-rom-license]"==0.23
pip3 install -U six
pip install -e .
cd ..

#flow is a dependency disaster of biblical scale
#git clone https://github.com/flow-project/flow.git
#cd flow
#python setup.py develop
#cd ..

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

conda deactivate && conda activate ./env

# this workflow only tests the libs
python -c "import gym, d4rl"

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestD4RL --error-for-skips
coverage combine
coverage xml -i
