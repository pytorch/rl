#!/usr/bin/env python3
"""Collect nightly workflow status and update the status dashboard.

This script queries GitHub API for nightly orchestrator workflow runs and updates
the historical status data stored on gh-pages.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# All nightly workflows tracked by the orchestrator.
# These are the child workflow files called via workflow_call.
NIGHTLY_WORKFLOWS = [
    "test-linux.yml",
    "test-linux-llm.yml",
    "test-linux-habitat.yml",
    "test-linux-tutorials.yml",
    "test-linux-sota.yml",
    "test-linux-libs.yml",
    "test-windows-optdepts.yml",
    "lint.yml",
    "docs.yml",
    "benchmarks.yml",
    "validate-test-partitioning.yml",
    "nightly_build.yml",
    "build-wheels-linux.yml",
    "build-wheels-m1.yml",
    "build-wheels-windows.yml",
    "build-wheels-aarch64-linux.yml",
]

# Display names for workflows
WORKFLOW_DISPLAY_NAMES = {
    "test-linux.yml": "Linux Tests",
    "test-linux-llm.yml": "LLM Tests",
    "test-linux-habitat.yml": "Habitat Tests",
    "test-linux-tutorials.yml": "Tutorials Tests",
    "test-linux-sota.yml": "SOTA Tests",
    "test-linux-libs.yml": "Libs Tests",
    "test-windows-optdepts.yml": "Windows Tests",
    "lint.yml": "Lint",
    "docs.yml": "Documentation",
    "benchmarks.yml": "Benchmarks",
    "validate-test-partitioning.yml": "Test Partitioning",
    "nightly_build.yml": "Nightly Build",
    "build-wheels-linux.yml": "Wheels (Linux)",
    "build-wheels-m1.yml": "Wheels (M1)",
    "build-wheels-windows.yml": "Wheels (Windows)",
    "build-wheels-aarch64-linux.yml": "Wheels (Aarch64)",
}

# Job name prefixes for each workflow (used in orchestrator jobs).
# When a workflow is called via workflow_call in the orchestrator,
# its jobs appear as "{prefix} / {job-name}" in the parent run.
WORKFLOW_JOB_PREFIXES = {
    "test-linux.yml": "test-linux",
    "test-linux-llm.yml": "test-linux-llm",
    "test-linux-habitat.yml": "test-linux-habitat",
    "test-linux-tutorials.yml": "test-linux-tutorials",
    "test-linux-sota.yml": "test-linux-sota",
    "test-linux-libs.yml": "test-linux-libs",
    "test-windows-optdepts.yml": "test-windows-optdepts",
    "lint.yml": "lint",
    "docs.yml": "docs",
    "benchmarks.yml": "benchmarks",
    "validate-test-partitioning.yml": "validate-test-partitioning",
    "nightly_build.yml": "nightly-build",
    "build-wheels-linux.yml": "build-wheels-linux",
    "build-wheels-m1.yml": "build-wheels-m1",
    "build-wheels-windows.yml": "build-wheels-windows",
    "build-wheels-aarch64-linux.yml": "build-wheels-aarch64-linux",
}


def get_run_jobs(
    owner: str,
    repo: str,
    run_id: int,
    token: str,
) -> list[dict]:
    """Fetch all jobs for a specific workflow run."""
    base_url = "https://api.github.com/repos"
    url = f"{base_url}/{owner}/{repo}/actions/runs/{run_id}/jobs"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {"per_page": 100}

    all_jobs = []
    while url:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        all_jobs.extend(data.get("jobs", []))

        # Handle pagination
        url = None
        if "next" in response.links:
            url = response.links["next"]["url"]
            params = {}  # Clear params for subsequent requests (URL includes them)

    return all_jobs


def get_orchestrator_runs(
    owner: str,
    repo: str,
    token: str,
    created_after: str,
    created_before: str,
) -> list[dict]:
    """Fetch orchestrator workflow runs within a date range."""
    base_url = "https://api.github.com/repos"
    workflow = "nightly_orchestrator.yml"
    url = f"{base_url}/{owner}/{repo}/actions/workflows/{workflow}/runs"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {
        "created": f"{created_after}..{created_before}",
        "per_page": 100,
        "event": "schedule",  # Only scheduled runs
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("workflow_runs", [])


def _get_workflow_status_from_jobs(jobs: list[dict], workflow_file: str) -> dict:
    """Determine workflow status from its jobs in the orchestrator run.

    When a workflow is called via workflow_call, its jobs appear in the parent
    orchestrator run with names like "{prefix} / {job-name}".
    """
    prefix = WORKFLOW_JOB_PREFIXES.get(workflow_file, "")
    display_name = WORKFLOW_DISPLAY_NAMES.get(workflow_file, workflow_file)

    # Find all jobs belonging to this workflow
    workflow_jobs = [j for j in jobs if j["name"].startswith(f"{prefix} /")]

    if not workflow_jobs:
        return {
            "status": "skipped",
            "conclusion": None,
            "url": None,
            "display_name": display_name,
        }

    # Determine overall status:
    # - If any job failed, workflow failed
    # - If all jobs succeeded, workflow succeeded
    # - If any job is still running, workflow is in_progress
    # - Otherwise use the most common conclusion
    conclusions = [j.get("conclusion") for j in workflow_jobs]
    statuses = [j.get("status") for j in workflow_jobs]

    if "failure" in conclusions:
        status = "failure"
    elif all(c == "success" for c in conclusions if c):
        status = "success"
    elif "in_progress" in statuses:
        status = "in_progress"
    elif all(c == "skipped" for c in conclusions if c):
        status = "skipped"
    elif "cancelled" in conclusions:
        status = "cancelled"
    else:
        # Use the first non-null conclusion
        status = next((c for c in conclusions if c), "unknown")

    # Get URL from the first job (they all link to the same run)
    job_url = workflow_jobs[0].get("html_url") if workflow_jobs else None

    return {
        "status": status,
        "conclusion": status if status != "in_progress" else None,
        "url": job_url,
        "display_name": display_name,
    }


def collect_status_for_date(
    owner: str,
    repo: str,
    token: str,
    date: datetime,
) -> dict:
    """Collect status for all nightly workflows for a specific date."""
    # Create date range for the given date (midnight to midnight UTC)
    date_str = date.strftime("%Y-%m-%d")
    created_after = f"{date_str}T00:00:00Z"
    created_before = f"{date_str}T23:59:59Z"

    # First, check if orchestrator ran on this date
    orchestrator_runs = get_orchestrator_runs(
        owner, repo, token, created_after, created_before
    )

    if not orchestrator_runs:
        return {
            "date": date_str,
            "orchestrator_ran": False,
            "orchestrator_status": "not_run",
            "orchestrator_url": None,
            "workflows": {},
        }

    # Get the latest orchestrator run for this date
    latest_orchestrator = orchestrator_runs[0]
    orchestrator_conclusion = latest_orchestrator["conclusion"]
    orchestrator_status = orchestrator_conclusion or latest_orchestrator["status"]
    orchestrator_run_id = latest_orchestrator["id"]

    # Fetch all jobs from the orchestrator run
    # Child workflows called via workflow_call appear as jobs in the parent run
    all_jobs = get_run_jobs(owner, repo, orchestrator_run_id, token)

    # Collect status for each workflow by analyzing its jobs
    workflows_status = {}
    for workflow_file in NIGHTLY_WORKFLOWS:
        workflows_status[workflow_file] = _get_workflow_status_from_jobs(
            all_jobs, workflow_file
        )

    return {
        "date": date_str,
        "orchestrator_ran": True,
        "orchestrator_status": orchestrator_status,
        "orchestrator_url": latest_orchestrator["html_url"],
        "orchestrator_run_id": orchestrator_run_id,
        "workflows": workflows_status,
    }


def calculate_summary(daily_status: dict) -> dict:
    """Calculate summary statistics for a day."""
    workflows = daily_status.get("workflows", {})
    if not workflows:
        return {
            "total": 0,
            "success": 0,
            "failure": 0,
            "skipped": 0,
            "success_rate": None,
        }

    total = len(workflows)
    success = sum(1 for w in workflows.values() if w["status"] == "success")
    failure = sum(1 for w in workflows.values() if w["status"] == "failure")
    skipped = sum(1 for w in workflows.values() if w["status"] == "skipped")
    # Count anything else as "other" (cancelled, in_progress, etc.)
    other = total - success - failure - skipped

    # Success rate is based on non-skipped workflows
    non_skipped = success + failure + other
    success_rate = (success / non_skipped * 100) if non_skipped > 0 else None

    rate_rounded = round(success_rate, 1) if success_rate is not None else None
    return {
        "total": total,
        "success": success,
        "failure": failure,
        "skipped": skipped,
        "other": other,
        "success_rate": rate_rounded,
    }


def load_existing_history(history_file: Path) -> list[dict]:
    """Load existing history from file."""
    if history_file.exists():
        with open(history_file) as f:
            data = json.load(f)
        return data.get("history", [])
    return []


def save_history(history: list[dict], output_file: Path) -> None:
    """Save history to file with metadata."""
    # Sort by date descending (newest first)
    history.sort(key=lambda x: x["date"], reverse=True)

    # Keep only last 90 days
    history = history[:90]

    # Calculate overall stats
    recent_7_days = history[:7]
    recent_30_days = history[:30]

    def calc_period_stats(days: list[dict]) -> dict:
        total_runs = sum(
            d["summary"]["success"]
            + d["summary"]["failure"]
            + d["summary"].get("other", 0)
            for d in days
            if d.get("orchestrator_ran")
        )
        total_success = sum(
            d["summary"]["success"] for d in days if d.get("orchestrator_ran")
        )
        total_failure = sum(
            d["summary"]["failure"] for d in days if d.get("orchestrator_ran")
        )
        runs_with_data = sum(1 for d in days if d.get("orchestrator_ran"))
        rate = round(total_success / total_runs * 100, 1) if total_runs > 0 else None
        return {
            "days": len(days),
            "runs_with_data": runs_with_data,
            "total_workflow_runs": total_runs,
            "total_success": total_success,
            "total_failure": total_failure,
            "success_rate": rate,
        }

    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "last_7_days": calc_period_stats(recent_7_days),
            "last_30_days": calc_period_stats(recent_30_days),
        },
        "history": history,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def generate_badge_json(history: list[dict], output_file: Path) -> None:
    """Generate shields.io endpoint JSON for the badge."""
    if not history:
        badge_data = {
            "schemaVersion": 1,
            "label": "nightly",
            "message": "no data",
            "color": "lightgrey",
        }
    else:
        # Use the most recent day with data
        latest = next((d for d in history if d.get("orchestrator_ran")), None)

        if not latest:
            badge_data = {
                "schemaVersion": 1,
                "label": "nightly",
                "message": "no runs",
                "color": "lightgrey",
            }
        else:
            summary = latest.get("summary", {})
            success = summary.get("success", 0)
            failure = summary.get("failure", 0)
            total = success + failure + summary.get("other", 0)

            if failure == 0 and total > 0:
                color = "brightgreen"
                message = f"{success}/{total} passing"
            elif failure > 0 and success > failure:
                color = "yellow"
                message = f"{failure}/{total} failing"
            elif failure > 0:
                color = "red"
                message = f"{failure}/{total} failing"
            else:
                color = "lightgrey"
                message = "no data"

            badge_data = {
                "schemaVersion": 1,
                "label": "nightly",
                "message": message,
                "color": color,
            }

    with open(output_file, "w") as f:
        json.dump(badge_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect nightly workflow status")
    parser.add_argument("--owner", default="pytorch", help="GitHub repository owner")
    parser.add_argument("--repo", default="rl", help="GitHub repository name")
    parser.add_argument(
        "--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to collect (for backfill)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nightly-status"),
        help="Output directory",
    )
    parser.add_argument(
        "--history-file", type=Path, help="Existing history file to merge with"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recollection even if data already exists for a date",
    )

    args = parser.parse_args()

    if not args.token:
        log.error("GITHUB_TOKEN environment variable or --token argument required")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing history
    history_file = args.history_file or args.output_dir / "history.json"
    existing_history = load_existing_history(history_file)
    existing_dates = {d["date"] for d in existing_history}

    # Collect status for the requested days
    today = datetime.now(timezone.utc)
    new_entries = []

    for days_ago in range(args.days):
        date = today - timedelta(days=days_ago)
        date_str = date.strftime("%Y-%m-%d")

        # Skip if we already have data for this date (unless it's today or --force)
        if date_str in existing_dates and days_ago > 0 and not args.force:
            log.info("Skipping %s - already have data", date_str)
            continue

        log.info("Collecting status for %s...", date_str)
        try:
            daily_status = collect_status_for_date(
                args.owner, args.repo, args.token, date
            )
            daily_status["summary"] = calculate_summary(daily_status)
            new_entries.append(daily_status)
            log.info("  Orchestrator ran: %s", daily_status["orchestrator_ran"])
            if daily_status["orchestrator_ran"]:
                log.info("  Summary: %s", daily_status["summary"])
        except Exception as e:
            log.error("  Error collecting status: %s", e)

    # Merge new entries with existing history
    # Replace entries for dates we just collected (in case of updates)
    new_dates = {e["date"] for e in new_entries}
    merged_history = [
        d for d in existing_history if d["date"] not in new_dates
    ] + new_entries

    # Save outputs
    output_history_file = args.output_dir / "history.json"
    save_history(merged_history, output_history_file)
    log.info("Saved history to %s", output_history_file)

    # Generate badge
    badge_file = args.output_dir / "badge.json"
    generate_badge_json(merged_history, badge_file)
    log.info("Saved badge data to %s", badge_file)


if __name__ == "__main__":
    main()
