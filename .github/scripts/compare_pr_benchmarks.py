from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path


MATCHED_METADATA_FIELDS = (
    "device",
    "pr_number",
    "base_sha",
    "head_sha",
    "runner",
    "image",
    "python_version",
    "system_environment_sha256",
    "dependency_lock_sha256",
    "dependency_environment_sha256",
    "benchmark_definitions_source_sha",
    "benchmark_definitions_sha256",
    "benchmark_command",
    "benchmark_environment",
    "run_id",
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        if compact:
            json.dump(value, file, separators=(",", ":"), sort_keys=True)
        else:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")


def _benchmark_name(benchmark: dict) -> str:
    return benchmark.get("fullname") or benchmark.get("name") or "<unknown>"


def _benchmark_duration(benchmark: dict) -> float | None:
    stats = benchmark.get("stats", {})
    median = stats.get("median")
    if median:
        return float(median)
    mean = stats.get("mean")
    if mean:
        return float(mean)
    ops = stats.get("ops")
    if ops:
        return 1.0 / float(ops)
    return None


def _benchmark_ops(benchmark: dict) -> float | None:
    duration = _benchmark_duration(benchmark)
    return None if duration is None else 1.0 / duration


def _benchmark_map(document: dict) -> dict[str, dict]:
    return {
        _benchmark_name(benchmark): benchmark
        for benchmark in document.get("benchmarks", [])
    }


def comparison_rows(baseline: dict, contender: dict) -> list[dict]:
    baseline_by_name = _benchmark_map(baseline)
    contender_by_name = _benchmark_map(contender)
    rows = []
    for name in sorted(set(baseline_by_name) & set(contender_by_name)):
        baseline_ops = _benchmark_ops(baseline_by_name[name])
        contender_ops = _benchmark_ops(contender_by_name[name])
        if not baseline_ops or contender_ops is None:
            continue
        rows.append(
            {
                "name": name,
                "baseline_ops": baseline_ops,
                "contender_ops": contender_ops,
                "change": (contender_ops / baseline_ops - 1.0) * 100.0,
            }
        )
    return rows


def needs_confirmation(
    change: float, reporting_threshold: float, threshold_margin: float
) -> bool:
    significant_change = abs(change) >= reporting_threshold
    near_reporting_threshold = (
        abs(abs(change) - reporting_threshold) <= threshold_margin
    )
    return significant_change or near_reporting_threshold


def aggregate_benchmark_documents(documents: list[dict]) -> dict:
    if not documents:
        raise ValueError("At least one benchmark document is required.")

    benchmark_maps = [_benchmark_map(document) for document in documents]
    expected_names = set(benchmark_maps[0])
    for index, benchmarks in enumerate(benchmark_maps[1:], start=2):
        names = set(benchmarks)
        if names != expected_names:
            missing = sorted(expected_names - names)
            unexpected = sorted(names - expected_names)
            raise RuntimeError(
                f"Confirmation repetition {index} has different benchmarks. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )

    result = copy.deepcopy(documents[0])
    aggregated_benchmarks = []
    for first_benchmark in documents[0].get("benchmarks", []):
        name = _benchmark_name(first_benchmark)
        durations = [
            _benchmark_duration(benchmarks[name]) for benchmarks in benchmark_maps
        ]
        if any(duration is None for duration in durations):
            raise RuntimeError(f"Confirmation repetitions lack timing data for {name}.")
        durations = [float(duration) for duration in durations]
        duration = statistics.median(durations)
        if len(durations) == 1:
            q1 = q3 = duration
            stddev = 0.0
        else:
            q1, _, q3 = statistics.quantiles(durations, n=4, method="inclusive")
            stddev = statistics.stdev(durations)

        benchmark = copy.deepcopy(first_benchmark)
        total_measurement_rounds = sum(
            int(benchmarks[name].get("stats", {}).get("rounds", 0))
            for benchmarks in benchmark_maps
        )
        benchmark["stats"] = {
            "data": durations,
            "iqr": q3 - q1,
            "max": max(durations),
            "mean": statistics.fmean(durations),
            "median": duration,
            "min": min(durations),
            "ops": 1.0 / duration,
            "q1": q1,
            "q3": q3,
            "rounds": len(durations),
            "stddev": stddev,
            "total": sum(durations),
        }
        extra_info = dict(benchmark.get("extra_info", {}))
        extra_info.update(
            {
                "confirmation_repetitions": len(durations),
                "confirmation_run_medians": durations,
                "confirmation_total_measurement_rounds": total_measurement_rounds,
            }
        )
        benchmark["extra_info"] = extra_info
        aggregated_benchmarks.append(benchmark)

    result["benchmarks"] = aggregated_benchmarks
    result["confirmation_repetitions"] = len(documents)
    return result


def _device_directories(raw_root: Path) -> list[Path]:
    return sorted(
        path.parent for path in raw_root.glob("*/baseline.json") if path.is_file()
    )


def _load_raw_pair(device_dir: Path) -> tuple[dict, dict, dict, dict]:
    baseline = _load_json(device_dir / "baseline.json")
    contender = _load_json(device_dir / "contender.json")
    baseline_names = set(_benchmark_map(baseline))
    contender_names = set(_benchmark_map(contender))
    if baseline_names != contender_names:
        missing_from_baseline = sorted(contender_names - baseline_names)
        missing_from_contender = sorted(baseline_names - contender_names)
        raise RuntimeError(
            f"{device_dir.name} benchmark sets differ. Missing from baseline: "
            f"{missing_from_baseline}; missing from contender: "
            f"{missing_from_contender}."
        )
    baseline_metadata = _load_json(device_dir / "baseline-metadata.json")
    contender_metadata = _load_json(device_dir / "contender-metadata.json")
    for field in MATCHED_METADATA_FIELDS:
        baseline_value = baseline_metadata.get(field)
        contender_value = contender_metadata.get(field)
        if baseline_value != contender_value:
            raise RuntimeError(
                f"{device_dir.name} benchmark metadata differs for {field}: "
                f"{baseline_value!r} != {contender_value!r}."
            )
    if baseline_metadata.get("revision") != "baseline":
        raise RuntimeError(f"Unexpected baseline metadata in {device_dir}.")
    if contender_metadata.get("revision") != "contender":
        raise RuntimeError(f"Unexpected contender metadata in {device_dir}.")
    if baseline_metadata.get("sha") != baseline_metadata.get("base_sha"):
        raise RuntimeError(f"Baseline SHA does not match the PR base in {device_dir}.")
    if contender_metadata.get("sha") != contender_metadata.get("head_sha"):
        raise RuntimeError(f"Contender SHA does not match the PR head in {device_dir}.")
    if baseline_metadata.get(
        "benchmark_definitions_source_sha"
    ) != baseline_metadata.get("base_sha"):
        raise RuntimeError(
            f"Benchmark definitions do not come from the PR base in {device_dir}."
        )
    return baseline, contender, baseline_metadata, contender_metadata


def create_confirmation_plan(
    raw_root: Path,
    plan_root: Path,
    matrix_output: Path,
    image_by_device: dict[str, str],
    reporting_threshold: float,
    threshold_margin: float,
) -> dict:
    device_dirs = _device_directories(raw_root)
    if not device_dirs:
        raise RuntimeError(f"No raw benchmark pairs were found under {raw_root}.")

    matrix = {"include": []}
    for device_dir in device_dirs:
        baseline, contender, baseline_metadata, _ = _load_raw_pair(device_dir)
        device = baseline_metadata["device"]
        if device not in image_by_device:
            raise RuntimeError(f"No pinned container image was supplied for {device}.")
        if baseline_metadata["image"] != image_by_device[device]:
            raise RuntimeError(
                f"{device} used {baseline_metadata['image']}, not the pinned image "
                f"{image_by_device[device]}."
            )
        rows = comparison_rows(baseline, contender)
        if not rows:
            raise RuntimeError(f"No comparable {device} benchmarks were found.")
        candidates = [
            row
            for row in rows
            if needs_confirmation(row["change"], reporting_threshold, threshold_margin)
        ]
        plan = {
            "comparison_statistic": "median duration",
            "confirmation_repetitions_per_revision": 2,
            "device": device,
            "image": image_by_device[device],
            "reporting_threshold_pct": reporting_threshold,
            "threshold_margin_pct": threshold_margin,
            "nodeids": [row["name"] for row in candidates],
            "pytest_nodeids": [
                row["name"].removeprefix("benchmark-definitions/benchmarks/")
                for row in candidates
            ],
            "initial_changes": {row["name"]: row["change"] for row in candidates},
        }
        _write_json(plan_root / f"{device}.json", plan)
        if candidates:
            matrix["include"].append(
                {"device": device, "image": image_by_device[device]}
            )
    _write_json(matrix_output, matrix, compact=True)
    return matrix


def _merge_confirmed(document: dict, confirmed: dict, names: set[str]) -> dict:
    confirmed_by_name = _benchmark_map(confirmed)
    missing = names - set(confirmed_by_name)
    if missing:
        raise RuntimeError(
            "Confirmation output is missing benchmarks: " + ", ".join(sorted(missing))
        )
    merged = dict(document)
    merged["benchmarks"] = [
        confirmed_by_name.get(_benchmark_name(benchmark), benchmark)
        if _benchmark_name(benchmark) in names
        else benchmark
        for benchmark in document.get("benchmarks", [])
    ]
    return merged


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def _summary_section(
    device: str,
    rows: list[dict],
    confirmed_names: set[str],
    reporting_threshold: float,
) -> list[str]:
    regressions = sum(row["change"] <= -reporting_threshold for row in rows)
    improvements = sum(row["change"] >= reporting_threshold for row in rows)
    lines = [
        f"### {device}",
        "",
        f"Compared {len(rows)} benchmarks: {regressions} regressions and "
        f"{improvements} improvements at the {reporting_threshold:g}% threshold. "
        f"Replaced {len(confirmed_names)} parallel measurements with sequential "
        "balanced same-runner confirmations.",
        "",
        "| Benchmark | Baseline median ops | PR median ops | Change | Measurement |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    selected = sorted(rows, key=lambda row: abs(row["change"]), reverse=True)[:50]
    for row in selected:
        measurement = (
            "balanced same runner" if row["name"] in confirmed_names else "parallel"
        )
        lines.append(
            f"| `{row['name']}` | {_format_number(row['baseline_ops'])} | "
            f"{_format_number(row['contender_ops'])} | {row['change']:+.2f}% | "
            f"{measurement} |"
        )
    lines.append("")
    return lines


def finalize_results(
    raw_root: Path,
    plan_root: Path,
    confirmation_root: Path,
    output_root: Path,
    summary_path: Path,
) -> None:
    summary_lines = ["## Final PR benchmark comparison", ""]
    device_dirs = _device_directories(raw_root)
    if not device_dirs:
        raise RuntimeError(f"No raw benchmark pairs were found under {raw_root}.")

    for device_dir in device_dirs:
        baseline, contender, baseline_metadata, _ = _load_raw_pair(device_dir)
        device = baseline_metadata["device"]
        plan = _load_json(plan_root / f"{device}.json")
        confirmed_names = set(plan["nodeids"])
        if confirmed_names:
            confirmation_dir = confirmation_root / device
            confirmed_baseline = _load_json(confirmation_dir / "baseline.json")
            confirmed_contender = _load_json(confirmation_dir / "contender.json")
            baseline = _merge_confirmed(baseline, confirmed_baseline, confirmed_names)
            contender = _merge_confirmed(
                contender, confirmed_contender, confirmed_names
            )

        rows = comparison_rows(baseline, contender)
        for row in rows:
            row["measurement"] = (
                "same-runner-confirmation"
                if row["name"] in confirmed_names
                else "parallel-runners"
            )
        output_dir = output_root / device
        metadata = {
            key: value
            for key, value in baseline_metadata.items()
            if key not in {"revision", "sha"}
        }
        metadata.update(
            {
                "confirmed_benchmarks": sorted(confirmed_names),
                "comparison_statistic": "median duration",
                "reporting_threshold_pct": plan["reporting_threshold_pct"],
                "threshold_margin_pct": plan["threshold_margin_pct"],
                "confirmation_order": [
                    "baseline-1",
                    "contender-1",
                    "contender-2",
                    "baseline-2",
                ],
                "confirmation_repetitions_per_revision": 2,
            }
        )
        _write_json(output_dir / "baseline.json", baseline)
        _write_json(output_dir / "contender.json", contender)
        _write_json(output_dir / "metadata.json", metadata)
        _write_json(output_dir / "comparison.json", {"benchmarks": rows})
        summary_lines.extend(
            _summary_section(
                device,
                rows,
                confirmed_names,
                float(plan["reporting_threshold_pct"]),
            )
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def _image_mapping(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        device, separator, image = value.partition("=")
        if not separator or not device or not image:
            raise ValueError(f"Expected DEVICE=IMAGE, got {value!r}.")
        mapping[device] = image
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--raw-root", required=True, type=Path)
    plan_parser.add_argument("--plan-root", required=True, type=Path)
    plan_parser.add_argument("--matrix-output", required=True, type=Path)
    plan_parser.add_argument("--image", action="append", required=True)
    plan_parser.add_argument("--reporting-threshold", type=float, default=5.0)
    plan_parser.add_argument("--threshold-margin", type=float, default=2.0)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--raw-root", required=True, type=Path)
    finalize_parser.add_argument("--plan-root", required=True, type=Path)
    finalize_parser.add_argument("--confirmation-root", required=True, type=Path)
    finalize_parser.add_argument("--output-root", required=True, type=Path)
    finalize_parser.add_argument("--summary", required=True, type=Path)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output", required=True, type=Path)
    aggregate_parser.add_argument("inputs", nargs="+", type=Path)

    args = parser.parse_args()
    if args.command == "plan":
        create_confirmation_plan(
            raw_root=args.raw_root,
            plan_root=args.plan_root,
            matrix_output=args.matrix_output,
            image_by_device=_image_mapping(args.image),
            reporting_threshold=args.reporting_threshold,
            threshold_margin=args.threshold_margin,
        )
    elif args.command == "finalize":
        finalize_results(
            raw_root=args.raw_root,
            plan_root=args.plan_root,
            confirmation_root=args.confirmation_root,
            output_root=args.output_root,
            summary_path=args.summary,
        )
    else:
        documents = [_load_json(path) for path in args.inputs]
        _write_json(args.output, aggregate_benchmark_documents(documents))


if __name__ == "__main__":
    main()
