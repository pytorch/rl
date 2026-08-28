#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU
export MUJOCO_GL=egl

python -m torch.utils.collect_env
git config --global --add safe.directory '*'

python -c 'import torch;t = torch.ones([2,2], device="cuda:0");print(t);print("tensor device:" + str(t.device))'
python -c "import mjlab; import mujoco; import mujoco_warp; import warp; print('mjlab:', mjlab.__version__ if hasattr(mjlab, '__version__') else 'unknown')"

root_dir="$(git rev-parse --show-toplevel)"
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-mjlab.json --json-report-indent=2"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs/test_mjlab.py ${json_report_args} --instafail -v --durations 200 --capture no --error-for-skips
coverage combine -q
coverage xml -i

python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
