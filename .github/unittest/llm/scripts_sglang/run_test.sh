#!/usr/bin/env bash

# Run SGLang-specific tests only.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

# Activate environment
source "${env_dir}/bin/activate"

apt-get update && apt-get install -y git gcc cmake
ln -s /usr/bin/swig3.0 /usr/bin/swig 2>/dev/null || true

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False

python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

# JSON report for flaky test tracking
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-sglang.json --json-report-indent=2"

# Run only SGLang-related tests
# Uses glob pattern to pick up all sglang test files that exist
pytest test/llm/test_sglang*.py \
    ${json_report_args} \
    -vvv \
    --instafail \
    --durations 600 \
    --capture no \
    --timeout=600 \
    --runslow

# Upload test results with metadata for flaky tracking
python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
