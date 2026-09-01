# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run and aggregate DreamerV3 DMC Walker learning curves."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from torchrl._utils import logger as torchrl_logger

CONFIG_PATH = Path(__file__).with_name("config_dmc_walker.yaml")
BASE_CONFIG_PATH = CONFIG_PATH.with_name("config.yaml")
# Set for each run below. A caller override would break the seed loop.
_RESERVED_OVERRIDES = ("env.seed", "logger.metrics_jsonl")


def _quantile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated ``q`` quantile of ``values``."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _override_key(override: str) -> str:
    """Return the config key a Hydra override addresses."""
    return override.split("=", 1)[0].lstrip("+~").strip()


def effective_config(overrides: Sequence[str] = ()) -> DictConfig:
    """Compose the walker preset as Hydra will, with the caller's overrides."""
    config = OmegaConf.merge(
        OmegaConf.load(BASE_CONFIG_PATH), OmegaConf.load(CONFIG_PATH)
    )
    dotlist = [override.lstrip("+") for override in overrides if "=" in override]
    if dotlist:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))
    return config


def episode_cycle(config: DictConfig) -> int:
    """Return the environment steps between episode-completion bursts.

    Workers run to the same time limit, so episodes finish one episode apart.
    """
    num_envs = config.collector.num_envs
    if config.collector.count_reset_records:
        # The driver axis also counts the reset record of each episode.
        return (config.env.max_episode_steps + 1) * num_envs
    return config.env.max_episode_steps * num_envs


def validate_window_size(window_size: int, overrides: Sequence[str] = ()) -> None:
    """Refuse, before the runs start, a window too narrow for one episode."""
    config = effective_config(overrides)
    cycle = episode_cycle(config)
    if window_size < cycle:
        raise ValueError(
            f"benchmark.window_size={window_size} is below the {cycle}-step "
            f"episode cycle ({config.collector.num_envs} envs x "
            f"{config.env.max_episode_steps}-step episodes), so most windows "
            "would hold no completed episode. Shorten collector.total_frames "
            "to run a smaller ablation, and leave the window alone."
        )


def benchmark_settings(overrides: Sequence[str] = ()) -> dict:
    """Read the ``benchmark`` block of the walker preset, overrides applied."""
    settings = effective_config(overrides).benchmark
    return OmegaConf.to_container(settings, resolve=True)


def reject_reserved_overrides(overrides: Sequence[str]) -> None:
    """Refuse overrides of the keys this script sets per run.

    Hydra takes the last of a duplicated key, so ``env.seed`` would train one
    trajectory and report it under every seed's name.
    """
    for override in overrides:
        key = _override_key(override)
        if key in _RESERVED_OVERRIDES:
            raise ValueError(
                f"{key} is set per run by this script and cannot be overridden. "
                "Use benchmark.seeds to choose the seeds and --output-dir to "
                "choose where their metrics land."
            )


def _read_run(path: Path) -> dict:
    """Fold one run's jsonl into the fields the aggregation needs."""
    episode_steps: list[int] = []
    episode_returns: list[float] = []
    summary: dict | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record["type"] == "train_episode":
            episode_steps.append(record["environment_steps"])
            episode_returns.append(record["score"])
        elif record["type"] == "summary":
            summary = record
    if summary is None:
        raise ValueError(
            f"{path} has no summary record; the run did not finish, so its "
            f"total step count is unknown."
        )
    return {
        "seed": summary["seed"],
        "total_environment_steps": summary["total_environment_steps"],
        "training_episode_steps": episode_steps,
        "training_episode_returns": episode_returns,
    }


def aggregate_runs(paths: Sequence[Path], window_size: int) -> dict:
    """Aggregate stochastic training returns into fixed-step median/IQR bands.

    Returns ``environment_steps`` with ``median_return``,
    ``lower_quartile_return``, ``upper_quartile_return`` and
    ``per_seed_window_median`` aligned to it, plus ``window_size`` and ``seeds``.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    runs = [_read_run(path) for path in paths]
    total_steps = min(run["total_environment_steps"] for run in runs)
    steps = list(range(window_size, total_steps + 1, window_size))
    if not steps:
        raise ValueError(f"Runs must contain at least {window_size} environment steps.")
    window_medians = []
    for run in runs:
        episode_steps = run["training_episode_steps"]
        episode_returns = run["training_episode_returns"]
        medians = []
        for stop in steps:
            start = stop - window_size
            values = [
                score
                for step, score in zip(episode_steps, episode_returns)
                if start < step <= stop
            ]
            if not values:
                raise ValueError(
                    f"Seed {run['seed']} has no completed training episode in "
                    f"the ({start}, {stop}] window."
                )
            medians.append(_quantile(values, 0.5))
        window_medians.append(medians)
    across_seeds = list(zip(*window_medians))
    return {
        "environment_steps": steps,
        "median_return": [_quantile(window, 0.5) for window in across_seeds],
        "lower_quartile_return": [_quantile(window, 0.25) for window in across_seeds],
        "upper_quartile_return": [_quantile(window, 0.75) for window in across_seeds],
        "per_seed_window_median": window_medians,
        "window_size": window_size,
        "seeds": [run["seed"] for run in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dmc_walker_runs"))
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides for the example. Those under benchmark.* also "
            f"override the {CONFIG_PATH.name} block this script reads."
        ),
    )
    args = parser.parse_args()

    reject_reserved_overrides(args.overrides)
    settings = benchmark_settings(args.overrides)
    seeds = settings["seeds"]
    window_size = settings["window_size"]
    minimum_final_return = settings["minimum_final_median_return"]
    validate_window_size(window_size, args.overrides)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train.py")
    metrics_paths = []
    for seed in seeds:
        metrics_jsonl_path = args.output_dir / f"seed_{seed}.jsonl"
        command = [
            sys.executable,
            str(script),
            "--config-name=config_dmc_walker",
            f"env.seed={seed}",
            f"logger.metrics_jsonl={metrics_jsonl_path}",
            "logger.output_plot=null",
            *args.overrides,
        ]
        torchrl_logger.info("Running DMC Walker seed %d", seed)
        subprocess.run(command, check=True)
        metrics_paths.append(metrics_jsonl_path)

    summary = aggregate_runs(metrics_paths, window_size=window_size)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    final_median = summary["median_return"][-1]
    if final_median < minimum_final_return:
        raise RuntimeError(
            "Final median DMC Walker return "
            f"{final_median:.1f} is below {minimum_final_return:.1f}."
        )
    torchrl_logger.info(
        "Saved DMC Walker median/IQR curve to %s (final median %.1f)",
        summary_path,
        final_median,
    )


if __name__ == "__main__":
    main()
