# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Summarize TorchRL collector overhead from a torch.profiler Chrome trace."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase:
    name: str
    pattern: str
    categories: tuple[str, ...] = ()


DEFAULT_PHASES = (
    Phase("profiler.step", r"^ProfilerStep#\d+$", ("user_annotation",)),
    Phase("collector.rollout.range", r"^Collector\.rollout$", ("user_annotation",)),
    Phase("collector.policy.range", r"^Collector\.policy$", ("user_annotation",)),
    Phase(
        "env.step_and_maybe_reset.range",
        r"^EnvBase\.step_and_maybe_reset$",
        ("user_annotation",),
    ),
    Phase(
        "collector.rollout.py",
        r"torchrl/collectors/_single.py\(\d+\): rollout$",
        ("python_function",),
    ),
    Phase(
        "env.step_and_maybe_reset.py",
        r"torchrl/envs/common.py\(\d+\): step_and_maybe_reset$",
        ("python_function",),
    ),
    Phase("env.step.py", r"torchrl/envs/common.py\(\d+\): step$", ("python_function",)),
    Phase(
        "transformed_env._step.py",
        r"torchrl/envs/transforms/_base.py\(\d+\): _step$",
        ("python_function",),
    ),
    Phase(
        "gym_like._step.py",
        r"torchrl/envs/gym_like.py\(\d+\): _step$",
        ("python_function",),
    ),
    Phase(
        "gymnasium.wrapper.step.py",
        r"gymnasium/wrappers/common.py\(\d+\): step$",
        ("python_function",),
    ),
    Phase(
        "collector._update_traj_ids.py",
        r"torchrl/collectors/_single.py\(\d+\): _update_traj_ids$",
        ("python_function",),
    ),
    Phase(
        "tensordict.stack.py",
        r"tensordict/.+_torch_func.py\(\d+\): _stack$",
        ("python_function",),
    ),
    Phase(
        "tensordict.to.py",
        r"tensordict/.+base.py\(\d+\): to$",
        ("python_function",),
    ),
    Phase("aten.copy", r"^aten::copy_$", ("cpu_op",)),
    Phase("aten.clone", r"^aten::clone$", ("cpu_op",)),
    Phase("cuda.launch", r"^(cudaLaunchKernel|cuLaunchKernel)$", ()),
    Phase("cuda.sync", r"^cuda.*Synchronize$", ()),
)

DERIVED_DELTAS = (
    (
        "collector bookkeeping range",
        "collector.rollout.range",
        ("collector.policy.range", "env.step_and_maybe_reset.range"),
    ),
    (
        "env wrapper envelope",
        "env.step_and_maybe_reset.py",
        ("backend_env.step.py",),
    ),
    ("gym_like wrapper", "gym_like._step.py", ("backend_env.step.py",)),
    (
        "transformed env wrapper",
        "transformed_env._step.py",
        ("gym_like._step.py",),
    ),
    ("EnvBase.step wrapper", "env.step.py", ("transformed_env._step.py",)),
    (
        "step_and_maybe_reset outer",
        "env.step_and_maybe_reset.py",
        ("env.step.py",),
    ),
    ("collector broad gap", "collector.rollout.py", ("backend_env.step.py",)),
)

GPU_CATEGORIES = (
    "kernel",
    "cuda_runtime",
    "cuda_driver",
    "gpu_memcpy",
    "gpu_memset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize TorchRL/TensorDict overhead from a torch.profiler "
            "Chrome trace produced by Collector.enable_profile."
        )
    )
    parser.add_argument("trace", type=Path, help="Path to a Chrome trace JSON file.")
    parser.add_argument(
        "--backend-step-pattern",
        default=r"isaaclab/envs/manager_based_rl_env.py\(\d+\): step$",
        help=(
            "Regex identifying the backend environment step. The default matches "
            "IsaacLab manager_based_rl_env.py step frames."
        ),
    )
    parser.add_argument(
        "--vector-steps",
        type=int,
        help=(
            "Number of vectorized environment steps represented by the trace. "
            "Defaults to the number of backend step events."
        ),
    )
    parser.add_argument(
        "--envs-per-vector-step",
        type=int,
        help="Number of parallel envs per vectorized step, used for per-frame costs.",
    )
    parser.add_argument(
        "--top",
        default=12,
        type=int,
        help="Number of top TorchRL/TensorDict Python functions to print.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path where a machine-readable summary is written.",
    )
    return parser.parse_args()


def load_events(path: Path) -> list[dict]:
    with path.open() as file:
        payload = json.load(file)
    return [
        event
        for event in payload.get("traceEvents", ())
        if event.get("ph") == "X" and "dur" in event and "name" in event
    ]


def matches(event: dict, pattern: re.Pattern, categories: tuple[str, ...]) -> bool:
    if categories and event.get("cat") not in categories:
        return False
    return bool(pattern.search(event["name"]))


def summarize(events: list[dict]) -> dict[str, float | int]:
    durations = [event["dur"] / 1000 for event in events]
    if not durations:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(durations),
        "total_ms": sum(durations),
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "max_ms": max(durations),
    }


def collect_phase_stats(
    events: list[dict], backend_step_pattern: str
) -> dict[str, dict[str, float | int]]:
    phases = (*DEFAULT_PHASES, Phase("backend_env.step.py", backend_step_pattern))
    stats = {}
    for phase in phases:
        pattern = re.compile(phase.pattern)
        stats[phase.name] = summarize(
            [event for event in events if matches(event, pattern, phase.categories)]
        )
    return stats


def collect_category_stats(events: list[dict]) -> dict[str, dict[str, float | int]]:
    return {
        category: summarize([event for event in events if event.get("cat") == category])
        for category in GPU_CATEGORIES
    }


def collect_top_python(
    events: list[dict], top: int
) -> list[tuple[str, int, float, float]]:
    by_name = {}
    for event in events:
        if event.get("cat") != "python_function":
            continue
        name = event["name"]
        if "torchrl/" not in name and "tensordict/" not in name:
            continue
        count, total_ms, max_ms = by_name.get(name, (0, 0.0, 0.0))
        duration_ms = event["dur"] / 1000
        by_name[name] = (
            count + 1,
            total_ms + duration_ms,
            max(max_ms, duration_ms),
        )
    return sorted(
        (
            (name, count, total_ms, max_ms)
            for name, (count, total_ms, max_ms) in by_name.items()
        ),
        key=lambda item: item[2],
        reverse=True,
    )[:top]


def format_ms(value: float | int) -> str:
    return f"{float(value):.3f}"


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def phase_rows(
    stats: dict[str, dict[str, float | int]], vector_steps: int, envs: int | None
) -> list[tuple[str, ...]]:
    rows = []
    for phase in (*DEFAULT_PHASES, Phase("backend_env.step.py", "")):
        phase_stats = stats[phase.name]
        total_ms = float(phase_stats["total_ms"])
        per_step_ms = total_ms / vector_steps if vector_steps else 0.0
        row = [
            phase.name,
            str(phase_stats["count"]),
            format_ms(total_ms),
            format_ms(phase_stats["mean_ms"]),
            format_ms(per_step_ms),
        ]
        if envs:
            row.append(format_ms(per_step_ms * 1000 / envs))
        rows.append(tuple(row))
    return rows


def derived_delta_stats(
    stats: dict[str, dict[str, float | int]], vector_steps: int, envs: int | None
) -> list[dict[str, float | int | str]]:
    rows = []
    for name, outer, inners in DERIVED_DELTAS:
        outer_ms = float(stats[outer]["total_ms"])
        inner_ms = sum(float(stats[inner]["total_ms"]) for inner in inners)
        total_ms = outer_ms - inner_ms
        per_step_ms = total_ms / vector_steps if vector_steps else 0.0
        row = {
            "name": name,
            "outer": outer,
            "inner": " + ".join(inners),
            "total_ms": total_ms,
            "ms_per_vector_step": per_step_ms,
        }
        if envs:
            row["us_per_env_frame"] = per_step_ms * 1000 / envs
        rows.append(row)
    return rows


def delta_rows(
    stats: dict[str, dict[str, float | int]], vector_steps: int, envs: int | None
) -> list[tuple[str, ...]]:
    rows = []
    for item in derived_delta_stats(stats, vector_steps, envs):
        row = [
            str(item["name"]),
            str(item["outer"]),
            str(item["inner"]),
            format_ms(float(item["total_ms"])),
            format_ms(float(item["ms_per_vector_step"])),
        ]
        if envs:
            row.append(format_ms(float(item["us_per_env_frame"])))
        rows.append(tuple(row))
    return rows


def category_rows(
    stats: dict[str, dict[str, float | int]], vector_steps: int
) -> list[tuple[str, ...]]:
    rows = []
    for category in GPU_CATEGORIES:
        category_stats = stats[category]
        total_ms = float(category_stats["total_ms"])
        rows.append(
            (
                category,
                str(category_stats["count"]),
                format_ms(total_ms),
                format_ms(category_stats["mean_ms"]),
                format_ms(total_ms / vector_steps if vector_steps else 0.0),
            )
        )
    return rows


def infer_vector_steps(stats: dict[str, dict[str, float | int]]) -> int:
    for phase in (
        "backend_env.step.py",
        "env.step_and_maybe_reset.range",
        "collector.policy.range",
    ):
        count = int(stats[phase]["count"])
        if count:
            return count
    return 0


def main() -> None:
    args = parse_args()
    events = load_events(args.trace)
    phase_stats = collect_phase_stats(events, args.backend_step_pattern)
    category_stats = collect_category_stats(events)
    vector_steps = args.vector_steps or infer_vector_steps(phase_stats)
    top_python = collect_top_python(events, args.top)

    print(f"Trace: {args.trace}")
    print(f"Complete events: {len(events)}")
    print(f"Vectorized env steps: {vector_steps}")
    if args.envs_per_vector_step:
        print(f"Env frames per vectorized step: {args.envs_per_vector_step}")
    print()

    headers = (
        "phase",
        "count",
        "total_ms",
        "mean_ms",
        "ms/vector_step",
    )
    if args.envs_per_vector_step:
        headers = (*headers, "us/env_frame")
    print("Inclusive Phase Times")
    print_table(
        headers, phase_rows(phase_stats, vector_steps, args.envs_per_vector_step)
    )
    print()

    delta_headers = ("overhead", "outer", "inner", "total_ms", "ms/vector_step")
    if args.envs_per_vector_step:
        delta_headers = (*delta_headers, "us/env_frame")
    print("Derived Overhead Estimates")
    print_table(
        delta_headers, delta_rows(phase_stats, vector_steps, args.envs_per_vector_step)
    )
    print()

    print("CUDA/GPU Categories")
    print_table(
        ("category", "count", "total_ms", "mean_ms", "ms/vector_step"),
        category_rows(category_stats, vector_steps),
    )
    print()

    print("Top TorchRL/TensorDict Python Functions")
    print_table(
        ("function", "count", "total_ms", "max_ms"),
        [
            (name, str(count), format_ms(total_ms), format_ms(max_ms))
            for name, count, total_ms, max_ms in top_python
        ],
    )

    if args.json_output:
        payload = {
            "trace": str(args.trace),
            "complete_events": len(events),
            "vector_steps": vector_steps,
            "envs_per_vector_step": args.envs_per_vector_step,
            "phases": phase_stats,
            "derived_overheads": derived_delta_stats(
                phase_stats, vector_steps, args.envs_per_vector_step
            ),
            "cuda_categories": category_stats,
            "top_python": [
                {
                    "name": name,
                    "count": count,
                    "total_ms": total_ms,
                    "max_ms": max_ms,
                }
                for name, count, total_ms, max_ms in top_python
            ],
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
