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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _benchmark_name(benchmark: dict) -> str:
    return benchmark.get("fullname") or benchmark.get("name") or "<unknown>"


def _benchmark_ops(benchmark: dict) -> float | None:
    """Return a rate based on the robust per-round duration statistic."""
    stats = benchmark.get("stats", {})
    median = stats.get("median")
    if median is not None and float(median) > 0:
        return 1.0 / float(median)
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


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def _summary_section(
    device: str, rows: list[dict], reporting_threshold: float
) -> list[str]:
    regressions = sum(row["change"] <= -reporting_threshold for row in rows)
    improvements = sum(row["change"] >= reporting_threshold for row in rows)
    lines = [
        f"### {device}",
        "",
        f"Compared {len(rows)} benchmarks: {regressions} regressions and "
        f"{improvements} improvements at the {reporting_threshold:g}% threshold. "
        "Each revision was measured once on a separate pinned runner.",
        "",
        "| Benchmark | Baseline ops | PR ops | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    selected = sorted(rows, key=lambda row: abs(row["change"]), reverse=True)[:50]
    for row in selected:
        lines.append(
            f"| `{row['name']}` | {_format_number(row['baseline_ops'])} | "
            f"{_format_number(row['contender_ops'])} | {row['change']:+.2f}% |"
        )
    lines.append("")
    return lines


def compare_results(
    raw_root: Path,
    output_root: Path,
    summary_path: Path,
    image_by_device: dict[str, str],
    reporting_threshold: float,
) -> None:
    summary_lines = [
        "## PR benchmark comparison",
        "",
        "Rates use inverse median round duration to limit isolated-stall bias.",
        "",
    ]
    device_dirs = _device_directories(raw_root)
    if not device_dirs:
        raise RuntimeError(f"No raw benchmark pairs were found under {raw_root}.")

    for device_dir in device_dirs:
        baseline, contender, baseline_metadata, _ = _load_raw_pair(device_dir)
        device = baseline_metadata["device"]
        expected_image = image_by_device.get(device)
        if expected_image is None:
            raise RuntimeError(f"No pinned container image was supplied for {device}.")
        if baseline_metadata["image"] != expected_image:
            raise RuntimeError(
                f"{device} used {baseline_metadata['image']}, not the pinned image "
                f"{expected_image}."
            )

        rows = comparison_rows(baseline, contender)
        if not rows:
            raise RuntimeError(f"No comparable {device} benchmarks were found.")
        for row in rows:
            row["measurement"] = "separate-pinned-runners"

        output_dir = output_root / device
        metadata = {
            key: value
            for key, value in baseline_metadata.items()
            if key not in {"revision", "sha"}
        }
        metadata.update(
            {
                "comparison_statistic": "median duration",
                "measurements_per_revision": 1,
                "reporting_threshold_pct": reporting_threshold,
            }
        )
        _write_json(output_dir / "baseline.json", baseline)
        _write_json(output_dir / "contender.json", contender)
        _write_json(output_dir / "metadata.json", metadata)
        _write_json(output_dir / "comparison.json", {"benchmarks": rows})
        summary_lines.extend(_summary_section(device, rows, reporting_threshold))

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
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--reporting-threshold", type=float, default=5.0)
    args = parser.parse_args()
    compare_results(
        raw_root=args.raw_root,
        output_root=args.output_root,
        summary_path=args.summary,
        image_by_device=_image_mapping(args.image),
        reporting_threshold=args.reporting_threshold,
    )


if __name__ == "__main__":
    main()
