#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

pytest test/smoke_test.py -v --durations 20
pytest test/smoke_test_deps.py -v --durations 20
pytest -v --durations 20
