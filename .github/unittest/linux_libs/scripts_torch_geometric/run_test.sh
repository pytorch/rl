#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

conda deactivate && conda activate ./env

python -c "import torch_geometric"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs/test_torch_geometric.py --instafail -v --durations 200 --capture no -k TestTorchGeometric --error-for-skips
coverage combine -q
coverage xml -i
