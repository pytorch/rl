#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# find libstdc
STDC_LOC=$(find conda/ -name "libstdc++.so.6" | head -1)

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export BATCHED_PIPE_TIMEOUT=60

MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest --instafail \
    -v --durations 200 --ignore test/test_distributed.py --ignore test/test_rlhf.py --capture no \
    --timeout=120 --mp_fork_if_no_cuda
coverage combine
coverage xml -i
