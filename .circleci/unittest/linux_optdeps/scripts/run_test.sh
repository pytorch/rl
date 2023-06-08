#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env
apt-get update && apt-get install -y git wget freeglut3 freeglut3-dev

# find libstdc
STDC_LOC=$(find conda/ -name "libstdc++.so.6" | head -1)

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch

#MUJOCO_GL=glfw pytest --cov=torchrl --junitxml=test-results/junit.xml -v --durations 200
MUJOCO_GL=egl python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest --instafail -v --durations 200 --ignore test/test_distributed.py
coverage combine
coverage xml -i
