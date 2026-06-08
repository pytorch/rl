from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from urllib import request


COMMENT_MARKER = "<!-- torchrl-pr-benchmark-comment -->"
MAX_COMMENT_LENGTH = 60000


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


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


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def _format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _truncate(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _collect_results(root: Path) -> list[dict]:
    results = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        result_dir = metadata_path.parent
        baseline_path = result_dir / "baseline.json"
        contender_path = result_dir / "contender.json"
        if not baseline_path.exists() or not contender_path.exists():
            continue
        metadata = _load_json(metadata_path)
        results.append(
            {
                "metadata": metadata,
                "baseline": _load_json(baseline_path),
                "contender": _load_json(contender_path),
            }
        )
    return results


def _comparison_rows(result: dict) -> list[dict]:
    baseline = {
        _benchmark_name(benchmark): benchmark
        for benchmark in result["baseline"].get("benchmarks", [])
    }
    contender = {
        _benchmark_name(benchmark): benchmark
        for benchmark in result["contender"].get("benchmarks", [])
    }
    rows = []
    for name in sorted(set(baseline) & set(contender)):
        baseline_ops = _benchmark_ops(baseline[name])
        contender_ops = _benchmark_ops(contender[name])
        if not baseline_ops or contender_ops is None:
            continue
        change = (contender_ops / baseline_ops - 1.0) * 100.0
        rows.append(
            {
                "name": name,
                "baseline_ops": baseline_ops,
                "contender_ops": contender_ops,
                "change": change,
            }
        )
    return rows


def _table(rows: list[dict], max_rows: int) -> list[str]:
    selected = sorted(rows, key=lambda row: abs(row["change"]), reverse=True)[:max_rows]
    lines = [
        "| Benchmark | main ops | PR ops | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                _truncate(row["name"]),
                _format_number(row["baseline_ops"]),
                _format_number(row["contender_ops"]),
                _format_percent(row["change"]),
            )
        )
    if len(rows) > max_rows:
        lines.append(
            f"| ... | ... | ... | Showing {max_rows} of {len(rows)} comparisons, sorted by absolute change. |"
        )
    return lines


def _device_section(result: dict, max_rows: int) -> list[str]:
    metadata = result["metadata"]
    rows = _comparison_rows(result)
    regressions = sum(row["change"] <= -5.0 for row in rows)
    improvements = sum(row["change"] >= 5.0 for row in rows)
    device = metadata["device"]
    lines = [
        f"#### {device}",
        "",
        f"Compared {len(rows)} benchmarks. Regressions over 5%: {regressions}. Improvements over 5%: {improvements}.",
        "",
    ]
    lines.extend(_table(rows, max_rows))
    lines.append("")
    return lines


def build_comment(results: list[dict], run_url: str) -> tuple[int, str]:
    if not results:
        raise RuntimeError("No benchmark artifacts were found.")
    pr_numbers = {int(result["metadata"]["pr_number"]) for result in results}
    if len(pr_numbers) != 1:
        raise RuntimeError(f"Expected one PR number, found {sorted(pr_numbers)}.")
    pr_number = pr_numbers.pop()
    first_metadata = results[0]["metadata"]
    base_sha = first_metadata["base_sha"][:8]
    head_sha = first_metadata["head_sha"][:8]
    max_rows = 120
    while max_rows >= 20:
        lines = [
            COMMENT_MARKER,
            f"### Benchmark Results: PR `{head_sha}` vs main `{base_sha}`",
            "",
            f"Benchmark run: {run_url}",
            "",
            "Higher ops/sec is better. Tables are sorted by largest absolute change.",
            "",
        ]
        for result in sorted(results, key=lambda item: item["metadata"]["device"]):
            lines.extend(_device_section(result, max_rows=max_rows))
        body = "\n".join(lines)
        if len(body) <= MAX_COMMENT_LENGTH:
            return pr_number, body
        max_rows -= 20
    return (
        pr_number,
        body[: MAX_COMMENT_LENGTH - 200]
        + "\n\nComment truncated due to GitHub size limits.\n",
    )


def _github_request(
    method: str, url: str, token: str, data: dict | None = None
) -> dict | list:
    body = None if data is None else json.dumps(data).encode()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    github_request = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(github_request) as response:
        response_body = response.read().decode()
    if not response_body:
        return {}
    return json.loads(response_body)


def upsert_comment(repository: str, pr_number: int, body: str) -> None:
    token = os.environ["GITHUB_TOKEN"]
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    comments_url = (
        f"{api_url}/repos/{repository}/issues/{pr_number}/comments?per_page=100"
    )
    comments = _github_request("GET", comments_url, token)
    for comment in comments:
        if COMMENT_MARKER in comment.get("body", ""):
            comment_url = (
                f"{api_url}/repos/{repository}/issues/comments/{comment['id']}"
            )
            _github_request("PATCH", comment_url, token, {"body": body})
            return
    create_url = f"{api_url}/repos/{repository}/issues/{pr_number}/comments"
    _github_request("POST", create_url, token, {"body": body})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--comment", action="store_true")
    args = parser.parse_args()

    results = _collect_results(args.artifact_root)
    pr_number, body = build_comment(results, args.run_url)
    print(body)
    if args.comment:
        upsert_comment(args.repository, pr_number, body)


if __name__ == "__main__":
    main()
