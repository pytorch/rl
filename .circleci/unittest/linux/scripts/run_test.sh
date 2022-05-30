#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [[ $OSTYPE == 'darwin'* ]]; then
  PRIVATE_MUJOCO_GL=glfw
else
  PRIVATE_MUJOCO_GL=glfw
fi

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
root_dir="$(git rev-parse --show-toplevel)"
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1
export MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210
export DISPLAY=unix:0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/project/.mujoco/mujoco210/bin

MUJOCO_GL=$PRIVATE_MUJOCO_GL pytest test/smoke_test.py -v --durations 20
MUJOCO_GL=$PRIVATE_MUJOCO_GL pytest test/smoke_test_deps.py -v --durations 20
MUJOCO_GL=$PRIVATE_MUJOCO_GL pytest -v --durations 20
