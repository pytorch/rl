# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run and aggregate reproducible DreamerV3 DMC Walker learning curves."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from torchrl._utils import logger as torchrl_logger


def aggregate_runs(paths: list[Path]) -> dict:
    """Aggregate aligned evaluation curves into median and interquartile bands."""
    runs = [json.loads(path.read_text()) for path in paths]
    steps = runs[0]["environment_steps"]
    if any(run["environment_steps"] != steps for run in runs[1:]):
        raise ValueError("All benchmark runs must use the same evaluation steps.")
    returns = torch.tensor(
        [run["evaluation_returns"] for run in runs], dtype=torch.float64
    )
    return {
        "environment_steps": steps,
        "median_return": returns.median(0).values.tolist(),
        "lower_quartile_return": torch.quantile(returns, 0.25, dim=0).tolist(),
        "upper_quartile_return": torch.quantile(returns, 0.75, dim=0).tolist(),
        "seeds": [run["seed"] for run in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("dmc_walker_runs"))
    parser.add_argument("--minimum-final-return", type=float, default=700.0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("dreamer_v3.py")
    metrics_paths = []
    for seed in args.seeds:
        metrics_path = args.output_dir / f"seed_{seed}.json"
        command = [
            sys.executable,
            str(script),
            "--config-name=config_dmc_walker",
            f"env.seed={seed}",
            f"logger.metrics_json={metrics_path}",
            "logger.output_plot=null",
            *args.overrides,
        ]
        torchrl_logger.info("Running DMC Walker seed %d", seed)
        subprocess.run(command, check=True)
        metrics_paths.append(metrics_path)

    summary = aggregate_runs(metrics_paths)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    final_median = summary["median_return"][-1]
    if final_median < args.minimum_final_return:
        raise RuntimeError(
            "Final median DMC Walker return "
            f"{final_median:.1f} is below {args.minimum_final_return:.1f}."
        )
    torchrl_logger.info(
        "Saved DMC Walker median/IQR curve to %s (final median %.1f)",
        summary_path,
        final_median,
    )


if __name__ == "__main__":
    main()
