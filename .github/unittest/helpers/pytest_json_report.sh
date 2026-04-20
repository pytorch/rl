#!/usr/bin/env bash
# Helper functions for pytest JSON report generation for flaky test tracking.
#
# Usage: Source this file in your test script before running pytest:
#   source .github/unittest/helpers/pytest_json_report.sh
#   pytest ... $(get_json_report_args "my-test-name")
#
# Or use the wrapper function:
#   run_pytest_with_json_report "my-test-name" test/test_foo.py -v --durations 200

# Get the directory for JSON report output
get_json_report_dir() {
    local root_dir
    root_dir="$(git rev-parse --show-toplevel 2>/dev/null || echo "${PWD}")"
    echo "${RUNNER_ARTIFACT_DIR:-${root_dir}}"
}

# Get pytest arguments for JSON report
# Usage: pytest ... $(get_json_report_args "test-name")
get_json_report_args() {
    local test_name="${1:-default}"
    local report_dir
    report_dir="$(get_json_report_dir)"
    echo "--json-report --json-report-file=${report_dir}/test-results-${test_name}.json --json-report-indent=2"
}

# Run pytest with JSON report and upload helper
# Usage: run_pytest_with_json_report "test-name" [pytest args...]
run_pytest_with_json_report() {
    local test_name="${1}"
    shift
    local json_args
    json_args="$(get_json_report_args "${test_name}")"
    
    # Run pytest with JSON report args
    # shellcheck disable=SC2086
    pytest ${json_args} "$@"
    local exit_code=$?
    
    return $exit_code
}

# Run coverage + pytest with JSON report
# Usage: run_coverage_pytest_with_json_report "test-name" [pytest args...]
run_coverage_pytest_with_json_report() {
    local test_name="${1}"
    shift
    local json_args
    json_args="$(get_json_report_args "${test_name}")"
    local root_dir
    root_dir="$(git rev-parse --show-toplevel 2>/dev/null || echo "${PWD}")"
    
    # Run with coverage wrapper if available
    if [ -f "${root_dir}/.github/unittest/helpers/coverage_run_parallel.py" ]; then
        # shellcheck disable=SC2086
        python "${root_dir}/.github/unittest/helpers/coverage_run_parallel.py" -m pytest ${json_args} "$@"
    else
        # shellcheck disable=SC2086
        pytest ${json_args} "$@"
    fi
    local exit_code=$?
    
    return $exit_code
}

# Call upload_test_results.py to add metadata
upload_test_results_with_metadata() {
    local root_dir
    root_dir="$(git rev-parse --show-toplevel 2>/dev/null || echo "${PWD}")"
    
    if [ -f "${root_dir}/.github/unittest/helpers/upload_test_results.py" ]; then
        python "${root_dir}/.github/unittest/helpers/upload_test_results.py" || \
            echo "Warning: Failed to process test results for flaky tracking"
    fi
}
