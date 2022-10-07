#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1
export MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210
export DISPLAY=unix:0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/project/.mujoco/mujoco210/bin
export MKL_THREADING_LAYER=GNU

#MUJOCO_GL=glfw pytest --cov=torchrl --junitxml=test-results/junit.xml -v --durations 20
MUJOCO_GL=egl coverage run -m pytest --instafail -v --durations 20
coverage xml
