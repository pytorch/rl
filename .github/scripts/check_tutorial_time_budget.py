# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fail the docs build when a sphinx-gallery example exceeds its time budget.

Sphinx-gallery executes every tutorial during the docs build. A single
heavyweight tutorial can multiply the build time without anyone noticing (the
MuJoCo macros tutorial landed at 45 minutes per build before this check
existed). This script parses the ``sg_execution_times.rst`` file that
sphinx-gallery writes (``write_computation_times: True`` in ``conf.py``) and
exits non-zero when any example exceeds the per-example budget.

Usage:
    python check_tutorial_time_budget.py [times_file] [--budget SECONDS]

The budget can also be set via the TORCHRL_TUTORIAL_TIME_BUDGET environment
variable (seconds); the --budget flag wins.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DEFAULT_TIMES_FILE = "docs/source/tutorials/sg_execution_times.rst"
DEFAULT_BUDGET_SECONDS = 420.0

# Matches list-table rows of the form:
#    * - :ref:`sphx_glr_tutorials_coding_dqn.py` (``coding_dqn.py``)
#      - 01:22.965
_ROW = re.compile(
    r"\(``(?P<name>[^`]+)``\)\s*\n\s*- (?P<time>[0-9:.]+)",
    re.MULTILINE,
)


def _to_seconds(stamp: str) -> float:
    """Convert ``[HH:]MM:SS.mmm`` to seconds."""
    parts = stamp.split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_times(text: str) -> dict[str, float]:
    return {m["name"]: _to_seconds(m["time"]) for m in _ROW.finditer(text)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("times_file", nargs="?", default=DEFAULT_TIMES_FILE)
    parser.add_argument(
        "--budget",
        type=float,
        default=float(
            os.environ.get("TORCHRL_TUTORIAL_TIME_BUDGET", DEFAULT_BUDGET_SECONDS)
        ),
        help="Per-example budget in seconds.",
    )
    args = parser.parse_args(argv)

    times_file = Path(args.times_file)
    if not times_file.exists():
        print(
            f"ERROR: {times_file} not found. Either the docs build did not run, "
            "sphinx-gallery stopped writing computation times "
            "(write_computation_times in docs/source/conf.py), or the gallery "
            "output moved. Update this check rather than deleting it."
        )
        return 1

    times = parse_times(times_file.read_text(encoding="utf-8"))
    if not times:
        print(
            f"ERROR: no execution times parsed from {times_file}. The "
            "sphinx-gallery output format may have changed; update the parser "
            "in this script."
        )
        return 1

    print(f"Per-example budget: {args.budget:.0f}s. Parsed {len(times)} examples:")
    over_budget = []
    for name, seconds in sorted(times.items(), key=lambda item: -item[1]):
        marker = " OVER BUDGET" if seconds > args.budget else ""
        print(f"  {seconds:8.1f}s  {name}{marker}")
        if seconds > args.budget:
            over_budget.append((name, seconds))

    if over_budget:
        print(
            f"\nERROR: {len(over_budget)} example(s) exceed the "
            f"{args.budget:.0f}s budget:"
        )
        for name, seconds in over_budget:
            print(f"  - {name}: {seconds:.1f}s")
        print(
            "Reduce the example's workload (fewer steps/rollouts, smaller "
            "renders, TORCHRL_TUTORIALS_FAST gating) or, if the cost is "
            "genuinely justified, raise TORCHRL_TUTORIAL_TIME_BUDGET in "
            ".github/workflows/docs.yml."
        )
        return 1

    print("All examples within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
