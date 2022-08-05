#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

pip3 install pyrender
pip3 install pyopengl --upgrade

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __GL_SHADER_DISK_CACHE=0
export __GL_SHADER_DISK_CACHE_PATH=/tmp
printf "DISPLAY:$DISPLAY-->\n"

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
printf "LD_LIBRARY_PATH:$LD_LIBRARY_PATH-->\n"
export MKL_THREADING_LAYER=GNU
export PATH=/home/circleci/project/env/bin:/home/circleci/project/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
printenv

pytest test/smoke_test.py -v --durations 20
pytest test/smoke_test_deps.py -v --durations 20
pytest --instafail -v --durations 20
