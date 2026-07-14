#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Track wall-clock duration of CI workflows over time.

This script:
1. Fetches recent workflow runs on main via the GitHub API
2. Computes wall-clock duration per run (first attempts only)
3. Merges them into a persistent history file (data.json on gh-pages)
4. Recomputes weekly median durations

Raw per-run records are retained for RUNS_RETENTION_DAYS; weekly medians
are kept forever, so the history file stays small. Run with a large
``--days`` value (e.g. 190) once to backfill.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


def log(msg: str = "") -> None:
    """Print with flush to ensure output is visible even if the runner dies."""
    print(msg, flush=True)


# =============================================================================
# Configuration
# =============================================================================

# Workflow file -> human-readable label shown on the dashboard.
# Wheel builds are intentionally not tracked.
TRACKED_WORKFLOWS: dict[str, str] = {
    "test-linux.yml": "Unit tests (Linux)",
    "docs.yml": "Doc build",
    "test-linux-libs.yml": "Libs tests",
    "test-windows-optdepts.yml": "Windows (optional deps)",
    "benchmarks.yml": "Benchmarks",
    "test-linux-sota.yml": "SOTA smoke tests",
    "test-linux-habitat.yml": "Habitat",
    "test-linux-llm.yml": "LLM tests",
    "test-linux-tutorials.yml": "Tutorials tests",
    "test-linux-mujoco.yml": "MuJoCo custom envs",
    "lint.yml": "Lint",
}

# Days of raw per-run records kept in the history file.
RUNS_RETENTION_DAYS = 120

# Sanity bounds on a single run's wall-clock duration (minutes).
MIN_DURATION_MIN = 0.5
MAX_DURATION_MIN = 24 * 60


# =============================================================================
# GitHub API helpers
# =============================================================================


def get_github_token() -> str:
    """Get GitHub token from environment."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        log("Error: GH_TOKEN or GITHUB_TOKEN environment variable required")
        sys.exit(1)
    return token


def get_repo() -> str:
    """Get repository from environment."""
    return os.environ.get("GITHUB_REPOSITORY", "pytorch/rl")


def _check_rate_limit(response: requests.Response) -> None:
    """Sleep if we're close to hitting the GitHub API rate limit."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) < 50:
        reset_at = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(0, reset_at - int(time.time())) + 5
        log(f"Rate limit low ({remaining} left), sleeping {wait}s")
        time.sleep(wait)


def fetch_workflow_runs(
    session: requests.Session, repo: str, workflow_file: str, since: datetime
) -> list[dict]:
    """Fetch completed first-attempt runs on main for one workflow."""
    runs: list[dict] = []
    url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/"
        f"{workflow_file}/runs"
    )
    params = {
        "branch": "main",
        "status": "completed",
        "created": f">={since.strftime('%Y-%m-%d')}",
        "per_page": 100,
        "page": 1,
    }
    while True:
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code == 404:
            log(f"  {workflow_file}: not found (404), skipping")
            return runs
        resp.raise_for_status()
        _check_rate_limit(resp)
        page_runs = resp.json().get("workflow_runs", [])
        runs.extend(page_runs)
        if len(page_runs) < params["per_page"]:
            return runs
        params["page"] += 1


def parse_ts(ts: str) -> datetime:
    """Parse a GitHub API UTC timestamp."""
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def run_to_record(run: dict) -> dict | None:
    """Convert an API run object to a compact history record, or None."""
    if run.get("run_attempt", 1) != 1:
        # Re-run attempts report misleading wall-clock durations.
        return None
    started, updated = run.get("run_started_at"), run.get("updated_at")
    if not started or not updated:
        return None
    duration_min = (parse_ts(updated) - parse_ts(started)).total_seconds() / 60
    if not MIN_DURATION_MIN <= duration_min <= MAX_DURATION_MIN:
        return None
    return {
        "id": run["id"],
        "date": started,
        "dur": round(duration_min, 1),
        "ok": run.get("conclusion") == "success",
        "event": run.get("event", ""),
    }


# =============================================================================
# History merge
# =============================================================================


def load_history(path: Path) -> dict:
    """Load an existing history file, or return an empty skeleton."""
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {"runs": {}, "weekly": {}}


def week_key(dt: datetime) -> str:
    """ISO week key, e.g. 2026-W23."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def merge_history(history: dict, new_records: dict[str, list[dict]]) -> dict:
    """Merge new run records into the history and refresh weekly medians."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=RUNS_RETENTION_DAYS)
    runs: dict[str, list[dict]] = history.get("runs", {})
    weekly: dict[str, dict[str, dict]] = history.get("weekly", {})

    for wf, records in new_records.items():
        by_id = {r["id"]: r for r in runs.get(wf, [])}
        for rec in records:
            by_id[rec["id"]] = rec
        merged = sorted(by_id.values(), key=lambda r: r["date"])
        # Refresh weekly medians for every week we still have raw runs for;
        # weeks older than the retention window keep their stored value.
        by_week: dict[str, list[float]] = {}
        for rec in merged:
            if rec["ok"]:
                by_week.setdefault(week_key(parse_ts(rec["date"])), []).append(
                    rec["dur"]
                )
        wf_weekly = weekly.setdefault(wf, {})
        for wk, durations in by_week.items():
            wf_weekly[wk] = {
                "median": round(statistics.median(durations), 1),
                "n": len(durations),
            }
        weekly[wf] = dict(sorted(wf_weekly.items()))
        runs[wf] = [r for r in merged if parse_ts(r["date"]) >= cutoff]

    return {
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_retention_days": RUNS_RETENTION_DAYS,
        "workflows": TRACKED_WORKFLOWS,
        "runs": runs,
        "weekly": weekly,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=4,
        help="Lookback window in days (use ~190 to backfill history)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Existing data.json to merge into (e.g. gh-pages/ci-timing/data.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ci-timing-out"),
        help="Directory to write the merged data.json to",
    )
    args = parser.parse_args()

    repo = get_repo()
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {get_github_token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    log(f"Fetching runs on main since {since:%Y-%m-%d} for {repo}")

    new_records: dict[str, list[dict]] = {}
    for wf in TRACKED_WORKFLOWS:
        api_runs = fetch_workflow_runs(session, repo, wf, since)
        records = [rec for run in api_runs if (rec := run_to_record(run))]
        new_records[wf] = records
        log(f"  {wf}: {len(api_runs)} runs fetched, {len(records)} kept")

    history = load_history(args.history) if args.history else {"runs": {}, "weekly": {}}
    merged = merge_history(history, new_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "data.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    n_runs = sum(len(v) for v in merged["runs"].values())
    n_weeks = sum(len(v) for v in merged["weekly"].values())
    log(f"Wrote {out_path} ({size_kb:.0f} kB, {n_runs} runs, {n_weeks} week entries)")


if __name__ == "__main__":
    main()
