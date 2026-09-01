#!/usr/bin/env bash

set -euo pipefail
set -x

root_dir="$(git rev-parse --show-toplevel)"
this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
mkdir -p "${json_report_dir}"
export TORCHRL_TEST_SHARD="${EXAMPLES_SHARD}"
export TORCHRL_TEST_SUITE=examples

python -m torch.utils.collect_env
pytest "${this_dir}/test_examples.py" \
  --json-report \
  --json-report-file="${json_report_dir}/test-results-examples-shard-${EXAMPLES_SHARD}.json" \
  --json-report-indent=2 \
  --instafail \
  --durations=200 \
  -vvv \
  --capture=no

python .github/unittest/helpers/upload_test_results.py || \
  echo "Warning: Failed to process example test results for flaky tracking"
