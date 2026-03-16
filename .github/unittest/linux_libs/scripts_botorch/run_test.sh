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

export MKL_THREADING_LAYER=GNU

# smoke test
python -c "import botorch; print('botorch', botorch.__version__)"
python -c "import gpytorch; print('gpytorch', gpytorch.__version__)"

# JSON report for flaky test tracking
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-botorch.json --json-report-indent=2"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_objectives.py ${json_report_args} --instafail -v --durations 200 --capture no -k TestGPWorldModel --error-for-skips
coverage combine -q
coverage xml -i

# Upload test results with metadata for flaky tracking
python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
