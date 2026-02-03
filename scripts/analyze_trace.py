#!/usr/bin/env python3
"""
PyTorch Profiler Trace Analysis Script

Usage:
    python3 scripts/analyze_trace.py <path_to_trace.json>

Example:
    python3 scripts/analyze_trace.py ./traces/merged_trace.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(trace_path: str) -> dict:
    """Load and validate a PyTorch profiler trace."""
    print(f"Loading trace from {trace_path}...")
    with open(trace_path) as f:
        trace = json.load(f)

    if "traceEvents" not in trace:
        raise ValueError("Invalid trace: missing traceEvents")

    return trace


def analyze_categories(events: list) -> dict:
    """Aggregate events by category."""
    cats = defaultdict(lambda: {"count": 0, "total_dur": 0})
    for e in events:
        if e.get("ph") == "X" and "dur" in e:
            cat = e.get("cat", "unknown")
            cats[cat]["count"] += 1
            cats[cat]["total_dur"] += e["dur"]
    return dict(cats)


def analyze_operations(events: list) -> dict:
    """Aggregate events by operation name."""
    ops = defaultdict(
        lambda: {
            "count": 0,
            "total_dur": 0,
            "min_dur": float("inf"),
            "max_dur": 0,
            "cat": None,
        }
    )
    for e in events:
        if e.get("ph") == "X" and "dur" in e:
            name = e.get("name", "unknown")
            dur = e["dur"]
            ops[name]["count"] += 1
            ops[name]["total_dur"] += dur
            ops[name]["min_dur"] = min(ops[name]["min_dur"], dur)
            ops[name]["max_dur"] = max(ops[name]["max_dur"], dur)
            ops[name]["cat"] = e.get("cat", "unknown")
    return dict(ops)


def analyze_gpu_utilization(events: list) -> dict:
    """Calculate GPU utilization metrics."""
    gpu_events = [
        e
        for e in events
        if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]
        and "ts" in e
        and "dur" in e
    ]

    if not gpu_events:
        return {"error": "No GPU events found"}

    gpu_events.sort(key=lambda x: x["ts"])

    first_ts = gpu_events[0]["ts"]
    last_end = gpu_events[-1]["ts"] + gpu_events[-1]["dur"]
    total_span = last_end - first_ts
    total_active = sum(e["dur"] for e in gpu_events)

    kernel_events = [e for e in gpu_events if e.get("cat") == "kernel"]
    kernel_time = sum(e["dur"] for e in kernel_events)

    return {
        "gpu_events": gpu_events,
        "total_events": len(gpu_events),
        "kernel_events": len(kernel_events),
        "timeline_span_us": total_span,
        "gpu_active_time_us": total_active,
        "kernel_time_us": kernel_time,
        "gpu_utilization_pct": 100 * total_active / total_span if total_span > 0 else 0,
        "compute_utilization_pct": 100 * kernel_time / total_span
        if total_span > 0
        else 0,
    }


def analyze_memory_transfers(events: list) -> dict:
    """Analyze GPU memory transfer patterns."""
    memcpy_events = [e for e in events if e.get("cat") == "gpu_memcpy" and "dur" in e]

    transfers = defaultdict(lambda: {"count": 0, "total_dur": 0, "total_bytes": 0})
    for e in memcpy_events:
        name = e.get("name", "unknown")
        transfers[name]["count"] += 1
        transfers[name]["total_dur"] += e["dur"]
        if "args" in e and "bytes" in e["args"]:
            transfers[name]["total_bytes"] += e["args"]["bytes"]

    return dict(transfers)


def analyze_user_annotations(events: list) -> dict:
    """Analyze user-defined code sections."""
    annotations = defaultdict(lambda: {"count": 0, "total_dur": 0, "gpu": False})
    for e in events:
        if e.get("cat") in ["user_annotation", "gpu_user_annotation"] and "dur" in e:
            name = e.get("name", "unknown")
            annotations[name]["count"] += 1
            annotations[name]["total_dur"] += e["dur"]
            annotations[name]["gpu"] = e.get("cat") == "gpu_user_annotation"
    return dict(annotations)


def find_gpu_gaps(gpu_events: list, min_gap_ms: float = 10.0) -> list:
    """Find gaps in GPU activity larger than threshold."""
    if not gpu_events:
        return []

    gaps = []
    for i in range(1, len(gpu_events)):
        prev_end = gpu_events[i - 1]["ts"] + gpu_events[i - 1]["dur"]
        curr_start = gpu_events[i]["ts"]
        gap_us = curr_start - prev_end

        if gap_us >= min_gap_ms * 1000:
            gaps.append(
                {
                    "gap_ms": gap_us / 1000,
                    "after_op": gpu_events[i - 1].get("name", "")[:50],
                    "after_cat": gpu_events[i - 1].get("cat", ""),
                    "before_op": gpu_events[i].get("name", "")[:50],
                    "before_cat": gpu_events[i].get("cat", ""),
                }
            )

    return sorted(gaps, key=lambda x: -x["gap_ms"])


def analyze_components(events: list) -> dict:
    """Analyze time distribution across components (for merged traces)."""
    components = defaultdict(lambda: {"count": 0, "total_dur": 0})
    for e in events:
        if e.get("ph") == "X" and "dur" in e:
            comp = e.get("args", {}).get("_component", "unknown")
            components[comp]["count"] += 1
            components[comp]["total_dur"] += e["dur"]
    return dict(components)


def detect_anomalies(ops: dict) -> list:
    """Detect operations with unexpectedly long durations."""
    expected_cheap = {
        "aten::stack": 500,
        "aten::cat": 500,
        "aten::index": 200,
        "aten::reshape": 50,
        "aten::view": 50,
        "aten::contiguous": 500,
        "aten::slice": 50,
        "aten::as_strided": 50,
        "aten::expand": 50,
        "aten::permute": 50,
        "aten::transpose": 50,
        "aten::squeeze": 50,
        "aten::unsqueeze": 50,
    }

    anomalies = []
    for name, info in ops.items():
        avg_dur = info["total_dur"] / info["count"] if info["count"] > 0 else 0

        for op_pattern, max_expected in expected_cheap.items():
            if op_pattern in name and avg_dur > max_expected:
                anomalies.append(
                    {
                        "operation": name,
                        "avg_duration_us": avg_dur,
                        "expected_max_us": max_expected,
                        "count": info["count"],
                        "total_time_s": info["total_dur"] / 1e6,
                        "severity": "HIGH" if avg_dur > max_expected * 10 else "MEDIUM",
                    }
                )

    return sorted(anomalies, key=lambda x: x.get("total_time_s", 0), reverse=True)


def print_report(trace: dict):
    """Print comprehensive analysis report."""
    events = trace["traceEvents"]
    metadata = trace.get("metadata", {})

    print("=" * 80)
    print("PYTORCH PROFILER TRACE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal events: {len(events):,}")
    print(f"Metadata: {metadata}")

    # Categories
    cats = analyze_categories(events)
    total_dur = sum(c["total_dur"] for c in cats.values())

    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    print(f"{'Category':<25} {'Count':>12} {'Duration (s)':>15} {'Percentage':>12}")
    print("-" * 70)
    for cat, info in sorted(cats.items(), key=lambda x: -x[1]["total_dur"]):
        pct = 100 * info["total_dur"] / total_dur if total_dur > 0 else 0
        print(
            f"{cat:<25} {info['count']:>12,} {info['total_dur']/1e6:>15.2f} {pct:>11.1f}%"
        )

    # Top operations
    ops = analyze_operations(events)

    print("\n" + "=" * 80)
    print("TOP 25 OPERATIONS BY TOTAL TIME")
    print("=" * 80)
    print(f"{'Operation':<55} {'Count':>10} {'Total (s)':>10} {'Avg (us)':>12}")
    print("-" * 92)
    for name, info in sorted(ops.items(), key=lambda x: -x[1]["total_dur"])[:25]:
        avg = info["total_dur"] / info["count"] if info["count"] > 0 else 0
        print(
            f"{name[:55]:<55} {info['count']:>10,} {info['total_dur']/1e6:>10.2f} {avg:>12.1f}"
        )

    # GPU utilization
    gpu_util = analyze_gpu_utilization(events)

    print("\n" + "=" * 80)
    print("GPU UTILIZATION")
    print("=" * 80)
    if "error" not in gpu_util:
        print(f"Timeline span:          {gpu_util['timeline_span_us']/1e6:.2f}s")
        print(
            f"GPU active time:        {gpu_util['gpu_active_time_us']/1e6:.2f}s ({gpu_util['gpu_utilization_pct']:.1f}%)"
        )
        print(
            f"Compute (kernel) time:  {gpu_util['kernel_time_us']/1e6:.2f}s ({gpu_util['compute_utilization_pct']:.1f}%)"
        )
        print(f"Total GPU events:       {gpu_util['total_events']:,}")
        print(f"Kernel events:          {gpu_util['kernel_events']:,}")
    else:
        print(f"Error: {gpu_util['error']}")

    # Memory transfers
    transfers = analyze_memory_transfers(events)

    print("\n" + "=" * 80)
    print("MEMORY TRANSFERS")
    print("=" * 80)
    print(f"{'Transfer Type':<50} {'Count':>10} {'Duration (s)':>12}")
    print("-" * 75)
    for name, info in sorted(transfers.items(), key=lambda x: -x[1]["total_dur"]):
        print(f"{name:<50} {info['count']:>10,} {info['total_dur']/1e6:>12.2f}")

    # Check for optimization opportunities
    for name, info in transfers.items():
        if "Pageable" in name and info["total_dur"] > 1e6:
            print(
                f"\n*** WARNING: {info['total_dur']/1e6:.2f}s in pageable transfers ({name})"
            )
            print("    -> Consider using pin_memory=True in DataLoader")

    # User annotations
    annotations = analyze_user_annotations(events)

    print("\n" + "=" * 80)
    print("USER ANNOTATIONS (code sections)")
    print("=" * 80)
    print(f"{'Section':<55} {'Count':>8} {'Duration (s)':>12}")
    print("-" * 80)
    for name, info in sorted(annotations.items(), key=lambda x: -x[1]["total_dur"])[
        :20
    ]:
        print(f"{name[:55]:<55} {info['count']:>8} {info['total_dur']/1e6:>12.2f}")

    # GPU gaps
    if "error" not in gpu_util:
        gaps = find_gpu_gaps(gpu_util["gpu_events"], min_gap_ms=10.0)

        print("\n" + "=" * 80)
        print(f"GPU IDLE GAPS > 10ms (total: {len(gaps)})")
        print("=" * 80)
        for g in gaps[:10]:
            print(
                f"{g['gap_ms']:>8.1f}ms gap: after '{g['after_op']}' -> before '{g['before_op']}'"
            )

    # Component breakdown
    components = analyze_components(events)

    print("\n" + "=" * 80)
    print("COMPONENT BREAKDOWN")
    print("=" * 80)
    for comp, info in sorted(components.items(), key=lambda x: -x[1]["total_dur"]):
        print(
            f"{comp:<25} {info['count']:>12,} events  {info['total_dur']/1e6:>12.2f}s"
        )

    # Anomaly detection
    anomalies = detect_anomalies(ops)

    if anomalies:
        print("\n" + "=" * 80)
        print("DETECTED ANOMALIES (unexpectedly slow operations)")
        print("=" * 80)
        for a in anomalies[:10]:
            print(f"[{a['severity']}] {a['operation'][:50]}")
            print(
                f"       Avg: {a['avg_duration_us']:.1f}us (expected <{a['expected_max_us']}us)"
            )
            print(f"       Total: {a['total_time_s']:.2f}s over {a['count']:,} calls")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the top operations - do they match expected workload?")
    print("2. Check GPU utilization - is it >70%? If not, why?")
    print("3. Look for anomalies - operations that shouldn't be slow")
    print("4. Check user annotations - is time spent where expected?")
    print("5. Review memory transfers - any pageable memory warnings?")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: No trace file specified")
        print("\nUsage: python3 scripts/analyze_trace.py <path_to_trace.json>")
        sys.exit(1)

    trace_path = sys.argv[1]

    if not Path(trace_path).exists():
        print(f"Error: File not found: {trace_path}")
        sys.exit(1)

    trace = load_trace(trace_path)
    print_report(trace)


if __name__ == "__main__":
    main()
