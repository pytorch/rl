"""Summarize repeated pytest measurements without counting warm-up as throughput."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(root: Path, summary: Path) -> None:
    paths = sorted(root.glob("async-*.json"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three completed repeats, found {len(paths)}")
    repeats = [json.loads(path.read_text()) for path in paths]
    indexed = [{row["fullname"]: row for row in run["benchmarks"]} for run in repeats]
    names = set(indexed[0])
    if not names or any(set(run) != names for run in indexed[1:]):
        raise RuntimeError(
            "The repeats must contain the same nonempty benchmark series"
        )
    lines = [
        "### Async environment benchmarks",
        "",
        "Three fresh processes, five measured rounds each; warm-up excluded. "
        "Ranges below describe the three run medians, not confidence intervals.",
        "",
        "| Series | Execution | Median frames/s | Run range | Batch p95 (ms) | RSS (MiB) | CUDA peak (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    trend = []
    for name in sorted(names):
        rows = [run[name] for run in indexed]
        executions = {row["extra_info"].get("execution", "pool") for row in rows}
        if len(executions) != 1:
            raise RuntimeError(f"Inconsistent execution configuration for {name}")
        execution = next(iter(executions))
        frames = {row["extra_info"]["transitions"] for row in rows}
        if len(frames) != 1 or next(iter(frames)) <= 0:
            raise RuntimeError(f"Inconsistent transition budget for {name}")
        fps = [
            row["extra_info"]["transitions"] / row["stats"]["median"] for row in rows
        ]
        median = statistics.median(fps)
        metrics = []
        for key, scale in [
            ("batch_latency_p95_ms", 1),
            ("process_tree_rss_bytes", 2**20),
            ("cuda_peak_allocated_bytes", 2**20),
        ]:
            values = [
                row["extra_info"][key] / scale
                for row in rows
                if key in row["extra_info"]
            ]
            metrics.append(f"{statistics.median(values):.2f}" if values else "-")
        spread = f"{min(fps):.1f}-{max(fps):.1f}"
        lines.append(
            f"| {name} | {execution} | {median:.1f} | {spread} | {' | '.join(metrics)} |"
        )
        trend.append(
            {
                "name": name,
                "unit": "frames/s",
                "value": median,
                "range": spread,
                "extra": f"Execution: {execution}. Batch p95: {metrics[0]} ms; process-tree RSS: {metrics[1]} MiB; CUDA peak: {metrics[2]} MiB. Three independent run medians.",
            }
        )
    lines.extend(
        [
            "",
            "RSS is a final process-tree snapshot, with shared pages counted per process. "
            "CUDA peak covers measured rounds when inference runs in this process. "
            "Raw samples, server timing, machine information and dependency versions are in the run artifacts.",
            "",
        ]
    )
    with summary.open("a") as stream:
        stream.write("\n".join(lines))
    (root / "trend.json").write_text(json.dumps(trend, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.summary)
