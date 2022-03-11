#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
root_dir="$(git rev-parse --show-toplevel)"
export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1
MUJOCO_GL=egl pytest  --cov=torchrl --junitxml=test-results/junit.xml -v --durations 20 --ignore third_party test
