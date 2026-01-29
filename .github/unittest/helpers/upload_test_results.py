# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Process pytest JSON results and add CI metadata for flaky test tracking.

This script:
1. Finds all test-results-*.json files in the artifact directory
2. Adds CI metadata (run ID, commit SHA, branch, etc.)
3. Combines them into a single file with all results
"""

import glob
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_git_info() -> dict:
    """Get git information for the current commit."""
    info = {}
    try:
        info["commit_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        info["commit_sha"] = os.environ.get("GITHUB_SHA", "unknown")

    try:
        info["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        info["branch"] = os.environ.get("GITHUB_REF_NAME", "unknown")

    return info


def get_ci_metadata() -> dict:
    """Collect CI metadata from environment variables."""
    git_info = get_git_info()

    return {
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "workflow_run_number": os.environ.get("GITHUB_RUN_NUMBER", "0"),
        "commit_sha": git_info["commit_sha"],
        "branch": git_info["branch"],
        "job_name": os.environ.get("GITHUB_JOB", "unknown"),
        "workflow_name": os.environ.get("GITHUB_WORKFLOW", "unknown"),
        "repository": os.environ.get("GITHUB_REPOSITORY", "pytorch/rl"),
        "event_name": os.environ.get("GITHUB_EVENT_NAME", "unknown"),
        "actor": os.environ.get("GITHUB_ACTOR", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
        "cuda_version": os.environ.get("CU_VERSION", "unknown"),
        "test_shard": os.environ.get("TORCHRL_TEST_SHARD", "all"),
        "test_suite": os.environ.get("TORCHRL_TEST_SUITE", "all"),
    }


def find_test_result_files(search_dir: Path) -> list[Path]:
    """Find all test-results-*.json files."""
    patterns = [
        search_dir / "test-results-*.json",
        search_dir / "**/test-results-*.json",
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern), recursive=True))
    return [Path(f) for f in set(files)]


def process_test_results(result_files: list[Path], metadata: dict) -> dict:
    """Combine test results from multiple files and add metadata."""
    combined = {
        "metadata": metadata,
        "shards": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "rerun": 0,
        },
        "tests": [],
    }

    for result_file in result_files:
        try:
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to read {result_file}: {e}")
            continue

        shard_name = result_file.stem.replace("test-results-", "")
        shard_info = {
            "name": shard_name,
            "file": str(result_file.name),
            "duration": data.get("duration", 0),
            "created": data.get("created", 0),
        }
        combined["shards"].append(shard_info)

        # Aggregate summary
        summary = data.get("summary", {})
        for key in [
            "passed",
            "failed",
            "error",
            "skipped",
            "xfailed",
            "xpassed",
            "rerun",
        ]:
            combined["summary"][key] += summary.get(key, 0)
        combined["summary"]["total_tests"] += summary.get("total", 0)

        # Collect individual test results
        tests = data.get("tests", [])
        for test in tests:
            test_entry = {
                "nodeid": test.get("nodeid", ""),
                "outcome": test.get("outcome", "unknown"),
                "duration": test.get("duration", 0),
                "shard": shard_name,
            }

            # Include setup/call/teardown durations if available
            for phase in ["setup", "call", "teardown"]:
                if phase in test:
                    phase_data = test[phase]
                    test_entry[f"{phase}_duration"] = phase_data.get("duration", 0)
                    if phase_data.get("outcome") == "failed":
                        test_entry[f"{phase}_crashed"] = True
                        # Include brief error info
                        longrepr = phase_data.get("longrepr", "")
                        if longrepr:
                            # Truncate long error messages
                            test_entry[f"{phase}_error"] = (
                                longrepr[:500] if len(longrepr) > 500 else longrepr
                            )

            # Track reruns (from pytest-rerunfailures)
            if "reruns" in test:
                test_entry["reruns"] = len(test["reruns"])
                test_entry["rerun_outcomes"] = [
                    r.get("outcome", "unknown") for r in test["reruns"]
                ]

            combined["tests"].append(test_entry)

    return combined


def main():
    # Determine directories
    root_dir = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
    artifact_dir = Path(os.environ.get("RUNNER_ARTIFACT_DIR", root_dir))

    print(f"Looking for test results in: {artifact_dir}")

    # Find test result files
    result_files = find_test_result_files(artifact_dir)

    # Also check root_dir if different from artifact_dir
    if artifact_dir != root_dir:
        result_files.extend(find_test_result_files(root_dir))
        result_files = list(set(result_files))

    if not result_files:
        print("No test result files found. Skipping flaky test tracking.")
        return

    print(f"Found {len(result_files)} test result files:")
    for f in result_files:
        print(f"  - {f}")

    # Get metadata
    metadata = get_ci_metadata()
    print(
        f"CI Metadata: run_id={metadata['workflow_run_id']}, sha={metadata['commit_sha'][:8]}"
    )

    # Process and combine results
    combined_results = process_test_results(result_files, metadata)

    # Write combined results
    output_file = artifact_dir / "test-results-combined.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Combined test results written to: {output_file}")
    print(f"Summary: {combined_results['summary']}")


if __name__ == "__main__":
    main()
