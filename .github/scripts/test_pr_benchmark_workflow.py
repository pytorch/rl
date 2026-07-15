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


def _document(ops_by_name: dict[str, float]) -> dict:
    return {
        "benchmarks": [
            {"fullname": name, "stats": {"ops": ops}}
            for name, ops in ops_by_name.items()
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
                    "test_bench.py::test_regression": 100.0,
                    "test_bench.py::test_near_threshold": 100.0,
                    "test_bench.py::test_clear_improvement": 100.0,
                }
            )
        ),
        encoding="utf-8",
    )
    (device_root / "contender.json").write_text(
        json.dumps(
            _document(
                {
                    "test_bench.py::test_regression": 90.0,
                    "test_bench.py::test_near_threshold": 104.0,
                    "test_bench.py::test_clear_improvement": 108.0,
                }
            )
        ),
        encoding="utf-8",
    )
    for revision in ("baseline", "contender"):
        (device_root / f"{revision}-metadata.json").write_text(
            json.dumps(_metadata("CPU", revision)), encoding="utf-8"
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [(-20.0, True), (-8.0, True), (-4.0, True), (4.0, True), (8.0, False)],
)
def test_confirmation_selection(change, expected):
    assert comparison.needs_confirmation(change, 5.0, 2.0) is expected


def test_plan_and_finalize_replace_only_confirmed_results(tmp_path):
    raw_root = tmp_path / "raw"
    plan_root = tmp_path / "plan"
    confirmation_root = tmp_path / "confirmation"
    output_root = tmp_path / "output"
    _write_raw_pair(raw_root)

    matrix = comparison.create_confirmation_plan(
        raw_root,
        plan_root,
        tmp_path / "matrix.json",
        {"CPU": "pinned-image"},
        reporting_threshold=5.0,
        threshold_margin=2.0,
    )
    assert matrix == {"include": [{"device": "CPU", "image": "pinned-image"}]}
    plan = json.loads((plan_root / "CPU.json").read_text(encoding="utf-8"))
    assert set(plan["nodeids"]) == {
        "test_bench.py::test_regression",
        "test_bench.py::test_near_threshold",
    }
    assert set(plan["pytest_nodeids"]) == set(plan["nodeids"])

    confirmation_device = confirmation_root / "CPU"
    confirmation_device.mkdir(parents=True)
    (confirmation_device / "baseline.json").write_text(
        json.dumps(
            _document(
                {
                    "test_bench.py::test_regression": 100.0,
                    "test_bench.py::test_near_threshold": 100.0,
                }
            )
        ),
        encoding="utf-8",
    )
    (confirmation_device / "contender.json").write_text(
        json.dumps(
            _document(
                {
                    "test_bench.py::test_regression": 98.0,
                    "test_bench.py::test_near_threshold": 106.0,
                }
            )
        ),
        encoding="utf-8",
    )

    comparison.finalize_results(
        raw_root,
        plan_root,
        confirmation_root,
        output_root,
        tmp_path / "summary.md",
    )
    result = json.loads(
        (output_root / "CPU" / "comparison.json").read_text(encoding="utf-8")
    )
    rows = {row["name"]: row for row in result["benchmarks"]}
    assert rows["test_bench.py::test_regression"]["change"] == pytest.approx(-2.0)
    assert rows["test_bench.py::test_near_threshold"]["change"] == pytest.approx(6.0)
    assert rows["test_bench.py::test_clear_improvement"]["change"] == pytest.approx(8.0)
    assert rows["test_bench.py::test_regression"]["measurement"] == (
        "same-runner-confirmation"
    )
    assert rows["test_bench.py::test_clear_improvement"]["measurement"] == (
        "parallel-runners"
    )

    final_metadata = json.loads(
        (output_root / "CPU" / "metadata.json").read_text(encoding="utf-8")
    )
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
    assert "Sequential same-runner confirmations: 2." in body
    assert "same runner" in body


def test_plan_uses_nodeids_relative_to_benchmark_working_directory(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_pair(raw_root)
    for revision in ("baseline", "contender"):
        result_path = raw_root / "CPU" / f"{revision}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for benchmark in result["benchmarks"]:
            benchmark["fullname"] = (
                "benchmark-definitions/benchmarks/" + benchmark["fullname"]
            )
        result_path.write_text(json.dumps(result), encoding="utf-8")

    comparison.create_confirmation_plan(
        raw_root,
        tmp_path / "plan",
        tmp_path / "matrix.json",
        {"CPU": "pinned-image"},
        reporting_threshold=5.0,
        threshold_margin=2.0,
    )

    plan = json.loads((tmp_path / "plan" / "CPU.json").read_text(encoding="utf-8"))
    assert all(
        nodeid.startswith("benchmark-definitions/benchmarks/")
        for nodeid in plan["nodeids"]
    )
    assert all(
        not nodeid.startswith("benchmark-definitions/benchmarks/")
        for nodeid in plan["pytest_nodeids"]
    )


def test_plan_rejects_mismatched_environments(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_pair(raw_root)
    metadata_path = raw_root / "CPU" / "contender-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["image"] = "different-image"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata differs for image"):
        comparison.create_confirmation_plan(
            raw_root,
            tmp_path / "plan",
            tmp_path / "matrix.json",
            {"CPU": "pinned-image"},
            reporting_threshold=5.0,
            threshold_margin=2.0,
        )


if __name__ == "__main__":
    pytest.main([__file__])
