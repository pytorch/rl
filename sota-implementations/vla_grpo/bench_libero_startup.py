# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark nested LIBERO ``ParallelEnv`` startup without loading a policy.

The benchmark mirrors the production five-by-sixty-four collector topology,
records every environment construction, executes one random outer step, and
verifies that shutdown leaves no environment processes behind.

Example:
    python sota-implementations/vla_grpo/bench_libero_startup.py \
        --mode worker-metadata --inner-start-method spawn \
        --output-dir /root/artifacts/libero-startup/worker-spawn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from queue import Empty

import torch
from omegaconf import OmegaConf
from torch import multiprocessing as mp
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.envs import ParallelEnv

_VLA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VLA_DIR))

import utils as vla_utils


def _instrumented_make_env_worker(
    cfg,
    tokenizer,
    worker_idx: int,
    *,
    marker_dir: str,
    subcollector_idx: int,
    **kwargs,
):
    parent_pid = int(os.environ["TORCHRL_LIBERO_BENCH_PARENT_PID"])
    role = "parent_metadata" if os.getpid() == parent_pid else "worker"
    marker_path = Path(marker_dir)
    marker_path.mkdir(parents=True, exist_ok=True)
    marker = marker_path / (
        f"construct-{role}-sub{subcollector_idx}-pid{os.getpid()}-"
        f"worker{worker_idx}"
    )
    marker.touch()
    return vla_utils._make_env_worker(
        cfg,
        tokenizer,
        worker_idx,
        **kwargs,
    )


def _make_nested_env(
    cfg,
    *,
    mode: str,
    inner_start_method: str,
    num_envs: int,
    subcollector_idx: int,
    marker_dir: str,
) -> ParallelEnv:
    worker_idx_offset = subcollector_idx * num_envs
    render_gpu_device_id = vla_utils._render_gpu_for_subcollector(cfg, subcollector_idx)
    factory = partial(
        _instrumented_make_env_worker,
        cfg,
        None,
        marker_dir=marker_dir,
        subcollector_idx=subcollector_idx,
        group_repeats=int(cfg.collector.group_size),
        seed=int(cfg.env.seed),
        device=None,
        worker_idx_offset=worker_idx_offset,
        render_gpu_device_id=render_gpu_device_id,
    )
    common_kwargs = {
        "mp_start_method": inner_start_method,
        "device": torch.device("cpu"),
    }
    if mode == "legacy-parent":
        return ParallelEnv(
            num_envs,
            [partial(factory, worker_idx=worker_idx) for worker_idx in range(num_envs)],
            **common_kwargs,
        )
    return ParallelEnv(
        num_envs,
        factory,
        create_env_kwargs=[
            {"worker_idx": worker_idx} for worker_idx in range(num_envs)
        ],
        metadata_from_workers=mode == "worker-metadata",
        **common_kwargs,
    )


def _subcollector_main(
    cfg_path: str,
    *,
    mode: str,
    inner_start_method: str,
    num_envs: int,
    subcollector_idx: int,
    marker_dir: str,
    status_queue,
    step_event,
) -> None:
    env = None
    os.environ["TORCHRL_LIBERO_BENCH_PARENT_PID"] = str(os.getpid())
    try:
        cfg = OmegaConf.load(cfg_path)
        construct_timer = timeit(
            f"libero_startup/subcollector_{subcollector_idx}/construct"
        ).start()
        env = _make_nested_env(
            cfg,
            mode=mode,
            inner_start_method=inner_start_method,
            num_envs=num_envs,
            subcollector_idx=subcollector_idx,
            marker_dir=marker_dir,
        )
        if env.is_closed:
            env.start()
        status_queue.put(
            {
                "event": "ready",
                "subcollector": subcollector_idx,
                "construct_s": construct_timer.elapsed(),
                "worker_pids": [process.pid for process in env._workers],
            }
        )
        if not step_event.wait(timeout=1800):
            raise TimeoutError("timed out waiting for the first-step barrier")

        step_timer = timeit(
            f"libero_startup/subcollector_{subcollector_idx}/first_step"
        ).start()
        reset = env.reset()
        step = env.rand_step(reset)
        instructions = reset.get("language_instruction")
        group_ids = reset.get("group_id")
        instruction_values = [str(instruction) for instruction in instructions]
        status_queue.put(
            {
                "event": "stepped",
                "subcollector": subcollector_idx,
                "first_step_s": step_timer.elapsed(),
                "reset_batch_size": list(reset.batch_size),
                "step_batch_size": list(step.batch_size),
                "instruction_count": len(instruction_values),
                "unique_instruction_count": len(set(instruction_values)),
                "group_ids": torch.as_tensor(group_ids).reshape(-1).tolist(),
            }
        )
    except Exception as err:
        status_queue.put(
            {
                "event": "error",
                "subcollector": subcollector_idx,
                "error": repr(err),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            env.close(raise_if_closed=False)
        status_queue.put(
            {
                "event": "closed",
                "subcollector": subcollector_idx,
            }
        )


def _descendant_pids(pid: int) -> set[int]:
    descendants = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(child) for child in children_path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            continue
        for child in children:
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, UnicodeDecodeError):
        return ""


def _collect_until(
    status_queue,
    *,
    event: str,
    count: int,
    processes,
    timeout_s: float,
    total_timer,
    messages: list[dict],
) -> tuple[list[dict], int]:
    selected = []
    selected_subcollectors = set()
    peak_processes = 0
    deadline_s = total_timer.elapsed() + timeout_s
    while len(selected) < count:
        if total_timer.elapsed() > deadline_s:
            raise TimeoutError(
                f"timed out waiting for {event!r} messages: "
                f"received {len(selected)}/{count}"
            )
        peak_processes = max(peak_processes, len(_descendant_pids(os.getpid())))
        try:
            message = status_queue.get(timeout=1.0)
        except Empty:
            dead = [
                process.pid
                for subcollector_idx, process in enumerate(processes)
                if subcollector_idx not in selected_subcollectors
                and not process.is_alive()
            ]
            if dead:
                raise RuntimeError(
                    f"outer subcollector processes exited before {event!r}: {dead}"
                )
            continue
        peak_processes = max(peak_processes, len(_descendant_pids(os.getpid())))
        messages.append(message)
        if message["event"] == "error":
            raise RuntimeError(
                f"subcollector {message['subcollector']} failed: "
                f"{message['error']}\n{message['traceback']}"
            )
        if message["event"] == event:
            selected.append(message)
            selected_subcollectors.add(message["subcollector"])
    return selected, peak_processes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("legacy-parent", "homogeneous-parent", "worker-metadata"),
        required=True,
    )
    parser.add_argument(
        "--inner-start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    parser.add_argument("--num-collectors", type=int, default=5)
    parser.add_argument("--envs-per-collector", type=int, default=64)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800,
        help="Maximum wait for each startup, step, and shutdown phase.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=_VLA_DIR / "config" / "vla_grpo_libero.yaml",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.inner_start_method != "spawn" and args.mode != "worker-metadata":
        raise ValueError(
            "forkserver/fork benchmarking is only allowed with worker-metadata, "
            "which guarantees that the subcollector parent has not created an "
            "EGL environment."
        )

    output_dir = args.output_dir.resolve()
    marker_dir = output_dir / "construction_markers"
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    (output_dir / "command.txt").write_text(command + "\n")

    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    step_event = ctx.Event()
    messages = []
    total_timer = timeit("libero_startup/total").start()
    processes = [
        ctx.Process(
            target=_subcollector_main,
            kwargs={
                "cfg_path": str(args.config.resolve()),
                "mode": args.mode,
                "inner_start_method": args.inner_start_method,
                "num_envs": args.envs_per_collector,
                "subcollector_idx": subcollector_idx,
                "marker_dir": str(marker_dir),
                "status_queue": status_queue,
                "step_event": step_event,
            },
        )
        for subcollector_idx in range(args.num_collectors)
    ]
    for process in processes:
        process.start()

    error = None
    peak_processes = 0
    ready = []
    stepped = []
    all_ready_s = None
    first_batch_s = None
    try:
        ready, peak = _collect_until(
            status_queue,
            event="ready",
            count=args.num_collectors,
            processes=processes,
            timeout_s=args.timeout_s,
            total_timer=total_timer,
            messages=messages,
        )
        peak_processes = max(peak_processes, peak)
        all_ready_s = total_timer.elapsed()
        step_event.set()
        stepped, peak = _collect_until(
            status_queue,
            event="stepped",
            count=args.num_collectors,
            processes=processes,
            timeout_s=args.timeout_s,
            total_timer=total_timer,
            messages=messages,
        )
        peak_processes = max(peak_processes, peak)
        first_batch_s = total_timer.elapsed()
        closed, peak = _collect_until(
            status_queue,
            event="closed",
            count=args.num_collectors,
            processes=processes,
            timeout_s=args.timeout_s,
            total_timer=total_timer,
            messages=messages,
        )
        del closed
        peak_processes = max(peak_processes, peak)
    except Exception:
        error = traceback.format_exc()
        step_event.set()
    finally:
        for process in processes:
            process.join(timeout=30)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()

    time.sleep(1)
    marker_names = [path.name for path in marker_dir.glob("construct-*")]
    parent_constructions = sum("parent_metadata" in name for name in marker_names)
    worker_constructions = sum("-worker-" in name for name in marker_names)
    remaining_descendants = sorted(_descendant_pids(os.getpid()))
    remaining_processes = {pid: _process_cmdline(pid) for pid in remaining_descendants}
    remaining_environment_processes = {
        pid: command
        for pid, command in remaining_processes.items()
        if "multiprocessing.resource_tracker" not in command
    }
    worker_pids = [pid for item in ready for pid in item["worker_pids"]]
    lingering_worker_processes = {
        pid: _process_cmdline(pid)
        for pid in worker_pids
        if Path(f"/proc/{pid}").exists()
    }
    summary = {
        "command": command,
        "mode": args.mode,
        "inner_start_method": args.inner_start_method,
        "num_collectors": args.num_collectors,
        "envs_per_collector": args.envs_per_collector,
        "total_envs": args.num_collectors * args.envs_per_collector,
        "parent_metadata_constructions": parent_constructions,
        "worker_constructions": worker_constructions,
        "subcollector_construct_s": [
            item["construct_s"]
            for item in sorted(ready, key=lambda item: item["subcollector"])
        ],
        "all_ready_s": all_ready_s,
        "subcollector_first_step_s": [
            item["first_step_s"]
            for item in sorted(stepped, key=lambda item: item["subcollector"])
        ],
        "first_batch_s": first_batch_s,
        "peak_descendant_processes": peak_processes,
        "outer_exit_codes": [process.exitcode for process in processes],
        "remaining_descendant_processes": remaining_processes,
        "remaining_environment_processes": remaining_environment_processes,
        "lingering_worker_processes": lingering_worker_processes,
        "messages": messages,
        "error": error,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torchrl_logger.info(json.dumps(summary, indent=2))
    if error is not None:
        raise RuntimeError(error)
    if parent_constructions != (
        args.num_collectors * args.envs_per_collector
        if args.mode == "legacy-parent"
        else args.num_collectors
        if args.mode == "homogeneous-parent"
        else 0
    ):
        raise RuntimeError(
            f"unexpected parent construction count: {parent_constructions}"
        )
    if worker_constructions != args.num_collectors * args.envs_per_collector:
        raise RuntimeError(
            f"unexpected worker construction count: {worker_constructions}"
        )
    if any(process.exitcode for process in processes):
        raise RuntimeError(f"outer process failures: {summary['outer_exit_codes']}")
    if remaining_environment_processes:
        raise RuntimeError(
            f"orphaned environment processes: {remaining_environment_processes}"
        )
    if lingering_worker_processes:
        raise RuntimeError(
            f"worker processes survived shutdown: {lingering_worker_processes}"
        )


if __name__ == "__main__":
    _main()
