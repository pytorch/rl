# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark ``ParallelEnv`` startup metadata strategies.

The benchmark records constructor calls in separate marker files so parent-side
temporary environments and long-lived worker environments can be counted
independently. For example::

    python benchmarks/benchmark_parallel_env_startup.py --workers 80 --delay 0.1

Use ``--mode`` to run one strategy, and ``--start-method forkserver`` to compare
the no-shadow-environment path with the default ``spawn`` worker startup.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Literal

from torchrl import timeit
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import ParallelEnv
from torchrl.testing.mocking_classes import CountingEnv


class _StartupEnv(CountingEnv):
    def __init__(
        self,
        worker_idx: int,
        *,
        marker_dir: str,
        parent_pid: int,
        delay: float,
    ) -> None:
        if delay:
            time.sleep(delay)
        role = "parent" if os.getpid() == parent_pid else "worker"
        self._marker_path = Path(marker_dir) / (
            f"constructed-{role}-{worker_idx}-{os.getpid()}"
        )
        self._marker_path.touch()
        super().__init__()

    def close(self, *, raise_if_closed: bool = True) -> None:
        self._marker_path.with_name(f"closed-{self._marker_path.name}").touch()
        super().close(raise_if_closed=raise_if_closed)


def _run_mode(
    mode: Literal["legacy", "homogeneous", "workers"],
    *,
    num_workers: int,
    delay: float,
    start_method: Literal["spawn", "forkserver", "fork"],
    marker_dir: Path,
) -> dict[str, float | int | str]:
    parent_pid = os.getpid()
    common_factory = partial(
        _StartupEnv,
        marker_dir=str(marker_dir),
        parent_pid=parent_pid,
        delay=delay,
    )
    create_env_kwargs = [
        {"worker_idx": worker_idx} for worker_idx in range(num_workers)
    ]
    if mode == "legacy":
        create_env_fn = [
            partial(common_factory, worker_idx=worker_idx)
            for worker_idx in range(num_workers)
        ]
        create_env_kwargs = None
    else:
        create_env_fn = common_factory

    with timeit(f"{mode}/construct") as construct_timer:
        env = ParallelEnv(
            num_workers,
            create_env_fn,
            create_env_kwargs=create_env_kwargs,
            metadata_from_workers=mode == "workers",
            use_buffers=False,
            mp_start_method=start_method,
        )
    construct_seconds = construct_timer.elapsed()
    try:
        with timeit(f"{mode}/reset") as reset_timer:
            tensordict = env.reset()
        reset_seconds = reset_timer.elapsed()
        with timeit(f"{mode}/first_step") as step_timer:
            env.rand_step(tensordict)
        first_step_seconds = step_timer.elapsed()
    finally:
        env.close(raise_if_closed=False)

    parent_constructions = len(list(marker_dir.glob("constructed-parent-*")))
    worker_constructions = len(list(marker_dir.glob("constructed-worker-*")))
    closed_constructions = len(list(marker_dir.glob("closed-constructed-*")))
    return {
        "mode": mode,
        "start_method": start_method,
        "num_workers": num_workers,
        "parent_constructions": parent_constructions,
        "worker_constructions": worker_constructions,
        "closed_constructions": closed_constructions,
        "construct_seconds": construct_seconds,
        "reset_seconds": reset_seconds,
        "first_step_seconds": first_step_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument(
        "--mode",
        choices=("all", "legacy", "homogeneous", "workers"),
        default="all",
    )
    parser.add_argument(
        "--start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    modes = ("legacy", "homogeneous", "workers") if args.mode == "all" else (args.mode,)
    results = []
    for mode in modes:
        with tempfile.TemporaryDirectory() as marker_dir:
            result = _run_mode(
                mode,
                num_workers=args.workers,
                delay=args.delay,
                start_method=args.start_method,
                marker_dir=Path(marker_dir),
            )
        results.append(result)
        torchrl_logger.info("ParallelEnv startup benchmark: %s", result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
