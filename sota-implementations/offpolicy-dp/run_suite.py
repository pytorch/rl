# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Launch the off-policy DP validations sequentially in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from torchrl._utils import logger as torchrl_logger


@dataclass(frozen=True)
class SuiteRun:
    algorithm: str
    profile: str


SMOKE_RUNS = (
    SuiteRun("dqn", "smoke_dqn"),
    SuiteRun("sac", "smoke_continuous"),
    SuiteRun("ddpg", "smoke_continuous"),
    SuiteRun("td3", "smoke_continuous"),
)

FULL_RUNS = (
    SuiteRun("dqn", "dqn"),
    SuiteRun("sac", "scale"),
    SuiteRun("ddpg", "scale"),
    SuiteRun("td3", "scale"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full", "all"),
        default="all",
        help="Run reduced smokes, planned full runs, or both in that order.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=("dqn", "sac", "ddpg", "td3"),
        help="Limit the suite to one or more algorithms.",
    )
    parser.add_argument(
        "--group",
        default="dp-stack-main",
        help="W&B group applied to every child run.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="/root/artifacts/offpolicy-dp",
        help="Remote directory for suite and child summaries.",
    )
    return parser.parse_args()


def _selected_runs(args: argparse.Namespace) -> list[SuiteRun]:
    runs: list[SuiteRun] = []
    if args.mode in ("smoke", "all"):
        runs.extend(SMOKE_RUNS)
    if args.mode in ("full", "all"):
        runs.extend(FULL_RUNS)
    if args.algorithm:
        selected = set(args.algorithm)
        runs = [run for run in runs if run.algorithm in selected]
    return runs


def main() -> int:
    args = _parse_args()
    script = Path(__file__).with_name("train.py")
    results = []
    for run in _selected_runs(args):
        command = [
            sys.executable,
            str(script),
            f"algorithm={run.algorithm}",
            f"profile={run.profile}",
            f"logging.group={args.group}",
            f"artifacts_dir={args.artifacts_dir}",
        ]
        torchrl_logger.info(f"Starting {run.algorithm} with profile={run.profile}.")
        completed = subprocess.run(command, check=False)
        results.append(
            {
                **asdict(run),
                "returncode": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
            }
        )
    artifact_dir = Path(args.artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary_path = artifact_dir / f"suite-{args.mode}-{timestamp}.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    torchrl_logger.info(f"Suite summary written to {summary_path}.")
    failures = [result for result in results if result["returncode"]]
    if failures:
        torchrl_logger.error(
            "Suite failures: "
            + ", ".join(f"{item['algorithm']}[{item['profile']}]" for item in failures)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
