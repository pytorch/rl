#!/usr/bin/env bash

set -e

conda activate torchrl

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

# we don't use torchsnapshot
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=60
export BATCHED_PIPE_TIMEOUT=60
export LAZY_LEGACY_OP=False

python -m torch.utils.collect_env
pytest --junitxml=test-results/junit.xml -v --durations 200  --ignore test/test_distributed.py --ignore test/test_rlhf.py
