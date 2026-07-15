from __future__ import annotations

import argparse
import json
import math
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


def _benchmark_ops(benchmark: dict) -> float | None:
    stats = benchmark.get("stats", {})
    ops = stats.get("ops")
    if ops is not None:
        return float(ops)
    mean = stats.get("mean")
    if mean:
        return 1.0 / float(mean)
    return None


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
    significant_regression = change <= -reporting_threshold
    near_reporting_threshold = (
        abs(abs(change) - reporting_threshold) <= threshold_margin
    )
    return significant_regression or near_reporting_threshold


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
        "same-runner confirmations.",
        "",
        "| Benchmark | Baseline ops | PR ops | Change | Measurement |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    selected = sorted(rows, key=lambda row: abs(row["change"]), reverse=True)[:50]
    for row in selected:
        measurement = "same runner" if row["name"] in confirmed_names else "parallel"
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
                "reporting_threshold_pct": plan["reporting_threshold_pct"],
                "threshold_margin_pct": plan["threshold_margin_pct"],
                "confirmation_order": ["baseline", "contender"],
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
    else:
        finalize_results(
            raw_root=args.raw_root,
            plan_root=args.plan_root,
            confirmation_root=args.confirmation_root,
            output_root=args.output_root,
            summary_path=args.summary,
        )


if __name__ == "__main__":
    main()
