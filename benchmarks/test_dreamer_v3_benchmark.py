# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Continuous benchmark of the DreamerV3 example's asynchronous training loop.

The example runs unmodified in a child process on a fixed fake pixel workload
(``bench_dreamer_v3_env.py``). Throughput is read from the example's own
metrics log, so setup, process startup and replay warm-up stay outside the
measured rounds. See ASYNC_BENCHMARKS.md ("DreamerV3 training series") before
changing any input.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import psutil
import pytest
import torch

_has_hydra = importlib.util.find_spec("hydra") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "sota-implementations" / "dreamer_v3"
BENCHMARK_DIR = Path(__file__).resolve().parent

ROUND_FRAMES = 1024
WARMUP_ROUNDS = 2
MEASURED_ROUNDS = 5
START_TIMEOUT_S = 180.0
ROUND_TIMEOUT_S = 120.0


def _benchmark_config(
    *,
    inference_backend: Literal["thread", "process"],
    train_ratio: float,
    device: str,
    metrics: Path,
):
    if not (_has_hydra and _has_omegaconf):
        pytest.skip("The DreamerV3 example requires hydra-core and omegaconf")
    from omegaconf import OmegaConf

    example_cfg = OmegaConf.load(EXAMPLE_DIR / "config.yaml")
    has_backend = "inference_backend" in example_cfg.collector
    if inference_backend != "thread" and not has_backend:
        pytest.skip(
            "collector.inference_backend is not available in the example on this revision"
        )
    cfg = OmegaConf.load(BENCHMARK_DIR / "dreamer_v3.yaml")
    if has_backend:
        cfg.collector.inference_backend = inference_backend
    if inference_backend == "process":
        cfg.collector.env_exchange = "auto"
    cfg.optimization.device = device
    cfg.optimization.train_ratio = train_ratio
    cfg.logger.metrics_jsonl = str(metrics)
    return cfg


class _MetricsTail:
    """Incrementally read the example's JSON-lines metrics file."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._partial = b""
        self.latest_train: dict | None = None

    def poll(self) -> dict | None:
        if not self.path.exists():
            return self.latest_train
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)
        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()
        for line in lines:
            if line.strip():
                record = json.loads(line)
                if record.get("type") == "train":
                    self.latest_train = record
        return self.latest_train


def _log_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _wait_for_train_record(tail, process, log_path, predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while True:
        record = tail.poll()
        if record is not None and predicate(record):
            return record
        if process.poll() is not None:
            raise RuntimeError(
                f"The example exited with code {process.returncode}:\n"
                f"{_log_tail(log_path)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"The example made no progress within {timeout_s} s:\n"
                f"{_log_tail(log_path)}"
            )
        time.sleep(0.002)


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    children = []
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        pass
    # The example finishes the current batch and shuts its collector down.
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass


def _run_training_benchmark(
    benchmark,
    *,
    inference_backend: Literal["thread", "process"],
    train_ratio: float,
    device: str,
    tmp_path: Path,
):
    metrics = tmp_path / "metrics.jsonl"
    cfg = _benchmark_config(
        inference_backend=inference_backend,
        train_ratio=train_ratio,
        device=device,
        metrics=metrics,
    )
    from omegaconf import OmegaConf

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)
    log_path = tmp_path / "example.log"
    tail = _MetricsTail(metrics)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(EXAMPLE_DIR / "train.py"),
                "--config-path",
                str(tmp_path),
                "--config-name",
                "config",
                f"hydra.run.dir={tmp_path}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(tmp_path),
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [str(BENCHMARK_DIR), os.environ.get("PYTHONPATH", "")]
                ),
            },
        )
    try:
        first = _wait_for_train_record(
            tail,
            process,
            log_path,
            lambda record: record["updates"] > 0,
            START_TIMEOUT_S,
        )
        start = first["environment_steps"] + WARMUP_ROUNDS * ROUND_FRAMES
        warm = _wait_for_train_record(
            tail,
            process,
            log_path,
            lambda record: record["environment_steps"] >= start,
            WARMUP_ROUNDS * ROUND_TIMEOUT_S,
        )
        progress = {"target": start + ROUND_FRAMES, "updates": warm["updates"]}
        measured_start = time.perf_counter()
        updates_before = progress["updates"]

        def measured_round():
            record = _wait_for_train_record(
                tail,
                process,
                log_path,
                lambda record: record["environment_steps"] >= progress["target"],
                ROUND_TIMEOUT_S,
            )
            progress["target"] += ROUND_FRAMES
            progress["updates"] = record["updates"]

        benchmark.pedantic(measured_round, rounds=MEASURED_ROUNDS, iterations=1)
        measured_seconds = time.perf_counter() - measured_start
        updates = progress["updates"] - updates_before
        rss = 0
        try:
            tree = psutil.Process(process.pid)
            rss = sum(
                member.memory_info().rss
                for member in [tree, *tree.children(recursive=True)]
            )
        except psutil.Error:
            pass
        benchmark.extra_info.update(
            execution=(
                f"eager learner on {device}; {inference_backend} inference; "
                f"train ratio {train_ratio:g}; {cfg.collector.num_envs} envs; "
                f"inference batch <= {cfg.collector.inference_max_batch_size}"
            ),
            num_envs=cfg.collector.num_envs,
            frames_per_batch=cfg.collector.frames_per_batch,
            transitions=ROUND_FRAMES,
            warmup_rounds=WARMUP_ROUNDS,
            measured_rounds=MEASURED_ROUNDS,
            learner_updates=updates,
            learner_updates_per_s=updates / measured_seconds,
            process_tree_rss_bytes=rss,
        )
    finally:
        _stop(process)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("train_ratio", [2, 16], ids=["ratio2", "ratio16"])
@pytest.mark.parametrize("inference_backend", ["thread", "process"])
def test_dreamer_v3_async_training(benchmark, inference_backend, train_ratio, tmp_path):
    """Fixed end-to-end training series; see ASYNC_BENCHMARKS.md before changing inputs."""
    _run_training_benchmark(
        benchmark,
        inference_backend=inference_backend,
        train_ratio=train_ratio,
        device="cuda:0",
        tmp_path=tmp_path,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
