# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Reproduce the recurrent-state comparison without selecting a public schema.

Run with the TensorDict schema PR on PYTHONPATH::

    python benchmarks/ad_hoc/bench_recurrent_state.py --output results.json

All container variants use identical parameters, tensor layouts and workloads.
The GTrXL candidate is a sequential reference, not an optimized window kernel.
"""
from __future__ import annotations

import argparse
import functools as ft
import itertools
import json
import operator
import platform
import tracemalloc
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torch.utils.benchmark import Timer

from torchrl.collectors import Collector
from torchrl.envs import Compose, InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.modules import set_recurrent_mode
from torchrl.testing._state_candidates import _make_candidate
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv


def _measure(fn, seconds):
    fn()
    result = Timer(stmt="fn()", globals={"fn": fn}, num_threads=1).blocked_autorange(
        min_run_time=seconds
    )
    return {
        "median_us": result.median * 1e6,
        "iqr_us": result.iqr * 1e6,
        "samples": len(result.raw_times),
    }


def _tensor_bytes(td):
    return sum(value.numel() * value.element_size() for value in td.values(True, True))


def _case(kind, container, batch, width, memory_len, window, device, seconds):
    torch.manual_seed(0)
    module = _make_candidate(
        kind, container, hidden_size=width, memory_len=memory_len, device=device
    )
    primer = module.make_tensordict_primer()
    base_env = ContinuousActionVecMockEnv(batch_size=[batch], device=device)
    # This mock builds batched specs but its base initializer drops batch_size.
    base_env.batch_size = torch.Size([batch])
    env = TransformedEnv(
        base_env,
        Compose(InitTracker(), primer),
    )
    policy = TensorDictSequential(
        module,
        TensorDictModule(
            nn.Linear(width, 7, device=device), in_keys=["embed"], out_keys=["action"]
        ),
    )
    collector = Collector(
        env,
        policy,
        frames_per_batch=batch * window,
        total_frames=-1,
        auto_register_policy_transforms=False,
    )
    iterator = iter(collector)
    try:
        rollout = next(iterator).clone()
        sample = rollout.clone()
        # A training slice starts from the carry before its first observation.
        sample["is_init"][:, 0] = True
        root = rollout[:, 0].exclude("next", "embed", "action").clone()
        step_data = policy(root.clone())
        state = (
            root.select(*primer.primers.keys())
            if container == "flat"
            else root.get(("agent", "state"))
        )
        source = dict(state.items())
        cls = type(state)
        leaf = next(iter(source))
        reset_primer = TensorDictPrimer(primer.primers.clone(), reset_key="_reset")
        reset_data = root.clone()
        reset_data["_reset"] = (torch.arange(batch, device=device) % 2 == 0).unsqueeze(
            -1
        )

        def reset():
            return reset_primer._reset(
                reset_data, TensorDict({}, [batch], device=device)
            )

        def policy_step():
            with torch.no_grad():
                return module(root.clone())

        def train_window():
            module.zero_grad(set_to_none=True)
            with set_recurrent_mode(True):
                output = module(sample.clone())
                output["embed"].square().mean().backward()

        operations = {
            "construct": ft.partial(
                cls.from_dict, source, batch_size=[batch], device=device
            ),
            "mapping_access": ft.partial(state.get, leaf),
            "typed_access": ft.partial(operator.itemgetter(leaf), state)
            if container in ("td", "flat")
            else ft.partial(operator.attrgetter(leaf), state),
            "spec_zero": primer.primers.zero,
            "partial_reset": reset,
            "step_mdp": ft.partial(env.step_mdp, step_data),
            "policy_step": policy_step,
            "collect": ft.partial(next, iterator),
            "train_window": train_window,
        }
        timing = {name: _measure(fn, seconds) for name, fn in operations.items()}
        timing["collect"]["frames_per_second"] = (
            batch * window * 1e6 / timing["collect"]["median_us"]
        )
        timing["train_window"]["frames_per_second"] = (
            batch * window * 1e6 / timing["train_window"]["median_us"]
        )
        # Python allocations are measured separately so tracing never affects timings.
        tracemalloc.start()
        policy_step()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        state_bytes = _tensor_bytes(state)
        return {
            "kind": kind,
            "container": container,
            "batch": batch,
            "hidden": width,
            "memory_len": memory_len if kind == "gtrxl" else None,
            "window": window,
            "timing": timing,
            "state_tensor_bytes": state_bytes,
            "rollout_tensor_bytes": _tensor_bytes(rollout),
            "rollout_state_tensor_bytes": 2 * window * state_bytes,
            "policy_step_python_peak_bytes": peak,
        }
    finally:
        collector.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sizes = [(1, 16, 8, 8)]
    if not args.small_only:
        sizes.append((32, 64, 64, 16))
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": args.device,
        "threads": 1,
        "cases": [],
        "notes": [
            "GTrXL uses a sequential reference window implementation.",
            "Payload bytes include both root and next-state snapshots.",
            "Python peak allocations exclude native tensor storage.",
        ],
    }
    for repeat, size, kind in itertools.product(
        range(args.repeats), sizes, ("gru", "lstm", "gtrxl")
    ):
        batch, width, memory_len, window = size
        containers = (
            ("td", "tc", "ttd") if kind == "gtrxl" else ("flat", "td", "tc", "ttd")
        )
        offset = repeat % len(containers)
        containers = containers[offset:] + containers[:offset]
        for container in containers:
            result = _case(
                kind,
                container,
                batch,
                width,
                memory_len,
                window,
                args.device,
                args.min_run_time,
            )
            result["repeat"] = repeat
            report["cases"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"run={repeat} {kind:5} {container:4} B={batch:2}: {result['timing']['collect']['frames_per_second']:.0f} frames/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
