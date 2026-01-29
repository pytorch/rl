#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Analyze test results from recent CI runs to identify flaky tests.

This script:
1. Fetches workflow run artifacts via GitHub API
2. Parses test result JSON files
3. Aggregates per-test statistics
4. Identifies flaky tests based on failure patterns
5. Generates JSON and Markdown reports
"""

import argparse
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# =============================================================================
# Configuration - Thresholds for flaky test detection
# =============================================================================

# Minimum failure rate to be considered flaky (below this = probably just fixed)
FLAKY_THRESHOLD_MIN = 0.05  # 5%

# Maximum failure rate to be considered flaky (above this = broken, not flaky)
FLAKY_THRESHOLD_MAX = 0.95  # 95%

# Minimum number of failures required to flag as flaky
MIN_FAILURES_FOR_FLAKY = 2

# Minimum number of executions required for analysis
MIN_EXECUTIONS = 3

# Days to consider a test "newly flaky"
NEW_FLAKY_DAYS = 7


# =============================================================================
# GitHub API Helpers
# =============================================================================


def get_github_token() -> str:
    """Get GitHub token from environment."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN environment variable required")
        sys.exit(1)
    return token


def get_repo() -> str:
    """Get repository from environment."""
    return os.environ.get("GITHUB_REPOSITORY", "pytorch/rl")


def github_api_request(endpoint: str, token: str) -> dict | list | None:
    """Make a GitHub API request."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com{endpoint}"

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 404:
        return None
    elif response.status_code == 403:
        # Rate limit or permission issue
        print(f"Warning: API request failed (403): {endpoint}")
        return None
    elif response.status_code != 200:
        print(f"Warning: API request failed ({response.status_code}): {endpoint}")
        return None

    return response.json()


def download_artifact(artifact_url: str, token: str) -> bytes | None:
    """Download an artifact zip file."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        artifact_url, headers=headers, timeout=120, allow_redirects=True
    )

    if response.status_code != 200:
        print(f"Warning: Failed to download artifact ({response.status_code})")
        return None

    return response.content


# =============================================================================
# Data Collection
# =============================================================================


def list_workflow_runs(
    repo: str, workflow_name: str, branch: str, num_runs: int, token: str
) -> list[dict]:
    """List recent workflow runs for a specific workflow on a branch."""
    runs = []
    page = 1
    per_page = min(100, num_runs)

    while len(runs) < num_runs:
        endpoint = f"/repos/{repo}/actions/workflows/{workflow_name}/runs?branch={branch}&status=completed&per_page={per_page}&page={page}"
        data = github_api_request(endpoint, token)

        if not data or "workflow_runs" not in data:
            break

        workflow_runs = data["workflow_runs"]
        if not workflow_runs:
            break

        runs.extend(workflow_runs)
        page += 1

        if len(workflow_runs) < per_page:
            break

    return runs[:num_runs]


def get_run_artifacts(repo: str, run_id: int, token: str) -> list[dict]:
    """Get artifacts for a specific workflow run."""
    endpoint = f"/repos/{repo}/actions/runs/{run_id}/artifacts"
    data = github_api_request(endpoint, token)

    if not data or "artifacts" not in data:
        return []

    return data["artifacts"]


def extract_test_results(artifact_content: bytes) -> list[dict]:
    """Extract test results from a downloaded artifact zip."""
    results = []

    try:
        with zipfile.ZipFile(io.BytesIO(artifact_content)) as zf:
            for filename in zf.namelist():
                if "test-results" in filename and filename.endswith(".json"):
                    try:
                        content = zf.read(filename)
                        data = json.loads(content)
                        results.append(data)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Failed to parse {filename}: {e}")
    except zipfile.BadZipFile:
        print("Warning: Invalid zip file")

    return results


def collect_test_data(
    repo: str, workflow_name: str, num_runs: int, token: str
) -> tuple[list[dict], dict]:
    """Collect test data from recent workflow runs."""
    print(f"Fetching last {num_runs} runs of {workflow_name} on main...")

    runs = list_workflow_runs(repo, workflow_name, "main", num_runs, token)
    print(f"Found {len(runs)} completed runs")

    all_test_data = []
    run_metadata = {}

    for run in runs:
        run_id = run["id"]
        run_date = run["created_at"]
        commit_sha = run["head_sha"]

        artifacts = get_run_artifacts(repo, run_id, token)

        # Look for test result artifacts
        test_artifacts = [
            a
            for a in artifacts
            if "test" in a["name"].lower() or "result" in a["name"].lower()
        ]

        # Also check default artifacts from pytorch/test-infra
        if not test_artifacts:
            test_artifacts = artifacts

        for artifact in test_artifacts:
            # Download artifact
            content = download_artifact(artifact["archive_download_url"], token)
            if not content:
                continue

            # Extract test results
            results = extract_test_results(content)
            for result in results:
                result["_run_id"] = run_id
                result["_run_date"] = run_date
                result["_commit_sha"] = commit_sha
                all_test_data.append(result)

        run_metadata[run_id] = {
            "date": run_date,
            "sha": commit_sha,
            "conclusion": run["conclusion"],
        }

    print(
        f"Collected {len(all_test_data)} test result files from {len(run_metadata)} runs"
    )
    return all_test_data, run_metadata


# =============================================================================
# Analysis
# =============================================================================


def aggregate_test_stats(test_data: list[dict]) -> dict[str, dict]:
    """Aggregate statistics per test across all runs."""
    test_stats = defaultdict(
        lambda: {
            "executions": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "reruns": 0,
            "total_duration": 0.0,
            "failure_dates": [],
            "run_ids": set(),
        }
    )

    for result in test_data:
        run_id = result.get("_run_id", "unknown")
        run_date = result.get("_run_date", "")

        # Handle both combined format and raw pytest-json-report format
        tests = result.get("tests", [])

        for test in tests:
            nodeid = test.get("nodeid", "")
            if not nodeid:
                continue

            outcome = test.get("outcome", "unknown")
            duration = test.get("duration", 0) or test.get("call_duration", 0) or 0

            stats = test_stats[nodeid]
            stats["executions"] += 1
            stats["run_ids"].add(run_id)
            stats["total_duration"] += duration

            if outcome == "passed":
                stats["passed"] += 1
            elif outcome == "failed":
                stats["failed"] += 1
                if run_date:
                    stats["failure_dates"].append(run_date)
            elif outcome == "error":
                stats["error"] += 1
                if run_date:
                    stats["failure_dates"].append(run_date)
            elif outcome == "skipped":
                stats["skipped"] += 1
            elif outcome == "xfailed":
                stats["xfailed"] += 1
            elif outcome == "xpassed":
                stats["xpassed"] += 1

            # Track reruns
            if test.get("reruns", 0) > 0:
                stats["reruns"] += test["reruns"]

    # Convert sets to lists for JSON serialization
    for _nodeid, stats in test_stats.items():
        stats["run_ids"] = list(stats["run_ids"])

    return dict(test_stats)


def calculate_flaky_score(stats: dict) -> float:
    """
    Calculate a flaky score for a test.

    Score is based on:
    - Failure rate (higher = more flaky, but capped at 50%)
    - Number of failures (more failures = more confident)
    - Presence of reruns (indicates retry failures)

    Returns a score between 0 and 1, where higher = more flaky.
    """
    if stats["executions"] < MIN_EXECUTIONS:
        return 0.0

    total_failures = stats["failed"] + stats["error"]
    failure_rate = total_failures / stats["executions"]

    # Tests that fail 100% or 0% are not flaky
    if failure_rate >= FLAKY_THRESHOLD_MAX or failure_rate <= FLAKY_THRESHOLD_MIN:
        return 0.0

    # Base score from failure rate (peak at 50%)
    if failure_rate <= 0.5:
        base_score = failure_rate * 2  # 0 to 1 as rate goes 0 to 0.5
    else:
        base_score = (1 - failure_rate) * 2  # 1 to 0 as rate goes 0.5 to 1

    # Confidence factor based on number of failures
    confidence = min(1.0, total_failures / 5)  # Full confidence at 5+ failures

    # Bonus for reruns (indicates test failed then passed on retry)
    rerun_bonus = min(0.2, stats["reruns"] * 0.05)

    return min(1.0, (base_score * confidence) + rerun_bonus)


def identify_flaky_tests(test_stats: dict[str, dict]) -> list[dict]:
    """Identify flaky tests based on statistics."""
    flaky_tests = []

    for nodeid, stats in test_stats.items():
        if stats["executions"] < MIN_EXECUTIONS:
            continue

        total_failures = stats["failed"] + stats["error"]

        if total_failures < MIN_FAILURES_FOR_FLAKY:
            continue

        failure_rate = total_failures / stats["executions"]

        if failure_rate <= FLAKY_THRESHOLD_MIN or failure_rate >= FLAKY_THRESHOLD_MAX:
            continue

        flaky_score = calculate_flaky_score(stats)
        if flaky_score <= 0:
            continue

        # Determine if newly flaky
        first_failure = None
        if stats["failure_dates"]:
            failure_dates = sorted(stats["failure_dates"])
            first_failure = failure_dates[0]

        is_new = False
        if first_failure:
            try:
                first_failure_dt = datetime.fromisoformat(
                    first_failure.replace("Z", "+00:00")
                )
                cutoff = datetime.now(timezone.utc) - timedelta(days=NEW_FLAKY_DAYS)
                is_new = first_failure_dt > cutoff
            except ValueError:
                pass

        flaky_tests.append(
            {
                "nodeid": nodeid,
                "executions": stats["executions"],
                "failures": total_failures,
                "failure_rate": round(failure_rate, 4),
                "flaky_score": round(flaky_score, 4),
                "passed": stats["passed"],
                "failed": stats["failed"],
                "error": stats["error"],
                "reruns": stats["reruns"],
                "avg_duration": round(stats["total_duration"] / stats["executions"], 3),
                "recent_failures": sorted(stats["failure_dates"])[-5:]
                if stats["failure_dates"]
                else [],
                "first_seen_flaky": first_failure,
                "is_new": is_new,
            }
        )

    # Sort by flaky score (highest first)
    flaky_tests.sort(key=lambda x: x["flaky_score"], reverse=True)

    return flaky_tests


# =============================================================================
# Report Generation
# =============================================================================


def generate_json_report(
    flaky_tests: list[dict],
    test_stats: dict[str, dict],
    run_metadata: dict,
    output_path: Path,
) -> dict:
    """Generate JSON report."""
    now = datetime.now(timezone.utc)

    # Calculate date range from runs
    if run_metadata:
        dates = [v["date"] for v in run_metadata.values()]
        start_date = min(dates)[:10] if dates else now.strftime("%Y-%m-%d")
        end_date = max(dates)[:10] if dates else now.strftime("%Y-%m-%d")
    else:
        start_date = end_date = now.strftime("%Y-%m-%d")

    new_flaky_count = sum(1 for t in flaky_tests if t.get("is_new", False))

    report = {
        "generated_at": now.isoformat(),
        "analysis_period": {
            "start": start_date,
            "end": end_date,
            "runs_analyzed": len(run_metadata),
        },
        "summary": {
            "total_tests": len(test_stats),
            "flaky_count": len(flaky_tests),
            "new_flaky_count": new_flaky_count,
            "resolved_count": 0,  # Would need historical comparison
        },
        "flaky_tests": flaky_tests,
        "thresholds": {
            "min_failure_rate": FLAKY_THRESHOLD_MIN,
            "max_failure_rate": FLAKY_THRESHOLD_MAX,
            "min_failures": MIN_FAILURES_FOR_FLAKY,
            "min_executions": MIN_EXECUTIONS,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def generate_markdown_report(report: dict, output_path: Path) -> None:
    """Generate Markdown report."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = report["summary"]
    flaky_tests = report["flaky_tests"]

    lines = [
        f"# Flaky Test Report - {now_str}",
        "",
        "## Summary",
        "",
        f"- **Flaky tests**: {summary['flaky_count']}",
        f"- **Newly flaky** (last 7 days): {summary['new_flaky_count']}",
        f"- **Resolved**: {summary['resolved_count']}",
        f"- **Total tests analyzed**: {summary['total_tests']}",
        f"- **CI runs analyzed**: {report['analysis_period']['runs_analyzed']}",
        "",
        "---",
        "",
    ]

    if flaky_tests:
        lines.extend(
            [
                "## Flaky Tests",
                "",
                "| Test | Failure Rate | Failures | Flaky Score | Last Failed |",
                "|------|--------------|----------|-------------|-------------|",
            ]
        )

        for test in flaky_tests[:20]:  # Top 20
            nodeid = test["nodeid"]
            # Shorten long nodeids
            if len(nodeid) > 60:
                nodeid = "..." + nodeid[-57:]

            rate_str = f"{test['failure_rate'] * 100:.1f}% ({test['failures']}/{test['executions']})"
            score_str = f"{test['flaky_score']:.2f}"
            last_failed = (
                test["recent_failures"][-1][:10] if test["recent_failures"] else "N/A"
            )

            new_marker = " 🆕" if test.get("is_new") else ""

            lines.append(
                f"| `{nodeid}`{new_marker} | {rate_str} | {test['failures']} | {score_str} | {last_failed} |"
            )

        lines.extend(["", ""])

        if summary["new_flaky_count"] > 0:
            lines.extend(["### Newly Flaky Tests", ""])
            new_tests = [t for t in flaky_tests if t.get("is_new")]
            for test in new_tests:
                lines.append(f"- `{test['nodeid']}`")
            lines.append("")
    else:
        lines.extend(
            [
                "## No Flaky Tests Detected! 🎉",
                "",
                "All tests are passing consistently.",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Configuration",
            "",
            f"- Minimum failure rate: {report['thresholds']['min_failure_rate'] * 100:.0f}%",
            f"- Maximum failure rate: {report['thresholds']['max_failure_rate'] * 100:.0f}%",
            f"- Minimum failures required: {report['thresholds']['min_failures']}",
            f"- Minimum executions required: {report['thresholds']['min_executions']}",
            "",
            "---",
            "",
            f"*Generated at {report['generated_at']}*",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_badge_json(flaky_count: int, output_path: Path) -> None:
    """Generate shields.io endpoint badge JSON."""
    if flaky_count == 0:
        color = "brightgreen"
        message = "0"
    elif flaky_count <= 5:
        color = "yellow"
        message = str(flaky_count)
    elif flaky_count <= 10:
        color = "orange"
        message = str(flaky_count)
    else:
        color = "red"
        message = str(flaky_count)

    badge = {
        "schemaVersion": 1,
        "label": "flaky tests",
        "message": message,
        "color": color,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(badge, f, indent=2)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze flaky tests from CI runs")
    parser.add_argument(
        "--runs", type=int, default=30, help="Number of runs to analyze per workflow"
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Single workflow file name (deprecated, use --workflows)",
    )
    parser.add_argument(
        "--workflows",
        default="test-linux.yml",
        help="Comma-separated list of workflow file names",
    )
    parser.add_argument(
        "--output-dir", default="flaky-reports", help="Output directory"
    )
    args = parser.parse_args()

    token = get_github_token()
    repo = get_repo()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse workflows - support both single --workflow and comma-separated --workflows
    if args.workflow:
        workflows = [args.workflow]
    else:
        workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]

    print(f"Analyzing flaky tests for {repo}")
    print(f"  Workflows: {', '.join(workflows)}")
    print(f"  Runs to analyze per workflow: {args.runs}")
    print()

    # Collect test data from all workflows
    all_test_data = []
    all_run_metadata = {}

    for workflow in workflows:
        print(f"\n{'=' * 60}")
        print(f"Processing workflow: {workflow}")
        print("=" * 60)

        test_data, run_metadata = collect_test_data(repo, workflow, args.runs, token)
        all_test_data.extend(test_data)
        all_run_metadata.update(run_metadata)

    if not all_test_data:
        print("\nNo test data collected from any workflow. Generating empty report.")
        test_stats = {}
        flaky_tests = []
    else:
        # Aggregate statistics
        print("\n" + "=" * 60)
        print("Aggregating test statistics across all workflows...")
        print("=" * 60)
        test_stats = aggregate_test_stats(all_test_data)
        print(f"Analyzed {len(test_stats)} unique tests")

        # Identify flaky tests
        print("Identifying flaky tests...")
        flaky_tests = identify_flaky_tests(test_stats)
        print(f"Found {len(flaky_tests)} flaky tests")

    # Generate reports
    print("\nGenerating reports...")

    json_report = generate_json_report(
        flaky_tests, test_stats, all_run_metadata, output_dir / "flaky-tests.json"
    )

    # Add workflow info to the report
    json_report["workflows_analyzed"] = workflows

    # Re-write with workflow info
    with open(output_dir / "flaky-tests.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    generate_markdown_report(json_report, output_dir / "flaky-tests.md")
    generate_badge_json(len(flaky_tests), output_dir / "badge.json")

    print(f"\nReports written to {output_dir}/")
    print("  - flaky-tests.json")
    print("  - flaky-tests.md")
    print("  - badge.json")

    # Set outputs for GitHub Actions
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"flaky_count={len(flaky_tests)}\n")
            f.write(f"new_flaky_count={json_report['summary']['new_flaky_count']}\n")
            f.write(f"resolved_count={json_report['summary']['resolved_count']}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
