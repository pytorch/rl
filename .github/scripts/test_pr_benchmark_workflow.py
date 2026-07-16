from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[2]


def _load_script(name: str):
    path = _REPO_ROOT / ".github" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = _load_script("compare_pr_benchmarks.py")
comments = _load_script("comment_pr_benchmarks.py")


def _document(durations_by_name: dict[str, float]) -> dict:
    return {
        "benchmarks": [
            {
                "fullname": name,
                "stats": {
                    "mean": duration,
                    "median": duration,
                    "ops": 1.0 / duration,
                },
            }
            for name, duration in durations_by_name.items()
        ]
    }


def _metadata(device: str, revision: str) -> dict:
    return {
        "device": device,
        "revision": revision,
        "sha": "base-sha" if revision == "baseline" else "head-sha",
        "pr_number": 123,
        "base_sha": "base-sha",
        "head_sha": "head-sha",
        "runner": "pinned-runner",
        "image": "pinned-image",
        "python_version": "3.10.20",
        "system_environment_sha256": "system-environment",
        "dependency_lock_sha256": "dependency-lock",
        "dependency_environment_sha256": "dependency-environment",
        "benchmark_definitions_source_sha": "base-sha",
        "benchmark_definitions_sha256": "benchmark-definitions",
        "benchmark_command": "pinned-command",
        "benchmark_environment": {"PYTHONHASHSEED": "0"},
        "run_id": "run-id",
    }


def _write_raw_pair(root: Path) -> None:
    device_root = root / "CPU"
    device_root.mkdir(parents=True)
    (device_root / "baseline.json").write_text(
        json.dumps(
            _document(
                {
                    "test_bench.py::test_regression": 0.010,
                    "test_bench.py::test_improvement": 0.010,
                }
            )
        ),
        encoding="utf-8",
    )
    (device_root / "contender.json").write_text(
        json.dumps(
            _document(
                {
                    "test_bench.py::test_regression": 1.0 / 90.0,
                    "test_bench.py::test_improvement": 1.0 / 108.0,
                }
            )
        ),
        encoding="utf-8",
    )
    for revision in ("baseline", "contender"):
        (device_root / f"{revision}-metadata.json").write_text(
            json.dumps(_metadata("CPU", revision)), encoding="utf-8"
        )


def test_comparison_prefers_median_over_mean_and_ops():
    baseline = {
        "benchmarks": [
            {
                "fullname": "test_bench.py::test_stall",
                "stats": {"median": 1.0, "mean": 1.0, "ops": 1.0},
            }
        ]
    }
    contender = {
        "benchmarks": [
            {
                "fullname": "test_bench.py::test_stall",
                "stats": {"median": 1.0, "mean": 10.0, "ops": 0.1},
            }
        ]
    }

    [row] = comparison.comparison_rows(baseline, contender)
    assert row["baseline_ops"] == pytest.approx(1.0)
    assert row["contender_ops"] == pytest.approx(1.0)
    assert row["change"] == pytest.approx(0.0)


def test_compare_writes_single_measurement_results(tmp_path):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    summary_path = tmp_path / "summary.md"
    _write_raw_pair(raw_root)

    comparison.compare_results(
        raw_root,
        output_root,
        summary_path,
        {"CPU": "pinned-image"},
        reporting_threshold=5.0,
    )

    result = json.loads(
        (output_root / "CPU" / "comparison.json").read_text(encoding="utf-8")
    )
    rows = {row["name"]: row for row in result["benchmarks"]}
    assert rows["test_bench.py::test_regression"]["change"] == pytest.approx(-10.0)
    assert rows["test_bench.py::test_improvement"]["change"] == pytest.approx(8.0)
    assert {row["measurement"] for row in rows.values()} == {"separate-pinned-runners"}

    final_metadata = json.loads(
        (output_root / "CPU" / "metadata.json").read_text(encoding="utf-8")
    )
    assert final_metadata["comparison_statistic"] == "median duration"
    assert final_metadata["measurements_per_revision"] == 1
    assert "confirmed_benchmarks" not in final_metadata
    summary = summary_path.read_text(encoding="utf-8")
    assert "once on a separate pinned runner" in summary

    _, body = comments.build_comment(
        [
            {
                "metadata": final_metadata,
                "baseline": json.loads(
                    (output_root / "CPU" / "baseline.json").read_text(encoding="utf-8")
                ),
                "contender": json.loads(
                    (output_root / "CPU" / "contender.json").read_text(encoding="utf-8")
                ),
            }
        ],
        "https://example.com/run",
    )
    assert "once on a separate pinned runner" in body
    assert "inverse median round duration" in body
    assert "same runner" not in body


def test_compare_rejects_mismatched_environments(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_pair(raw_root)
    metadata_path = raw_root / "CPU" / "contender-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["image"] = "different-image"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata differs for image"):
        comparison.compare_results(
            raw_root,
            tmp_path / "output",
            tmp_path / "summary.md",
            {"CPU": "pinned-image"},
            reporting_threshold=5.0,
        )


def test_compare_rejects_unpinned_image(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_pair(raw_root)

    with pytest.raises(RuntimeError, match="not the pinned image"):
        comparison.compare_results(
            raw_root,
            tmp_path / "output",
            tmp_path / "summary.md",
            {"CPU": "different-image"},
            reporting_threshold=5.0,
        )


def test_workflow_runs_only_four_primary_measurements():
    workflow = (_REPO_ROOT / ".github" / "workflows" / "benchmarks_pr.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("          - device: CPU") == 2
    assert workflow.count("          - device: GPU") == 2
    assert workflow.count("            revision: baseline") == 2
    assert workflow.count("            revision: contender") == 2
    assert "\n  confirmation:" not in workflow
    assert "\n  finalize:" not in workflow
    assert "same-runner" not in workflow
    assert "needs: benchmark" in workflow


if __name__ == "__main__":
    pytest.main([__file__])
