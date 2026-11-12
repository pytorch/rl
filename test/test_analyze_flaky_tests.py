from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

_SCRIPT_PATH = (
    Path(__file__).parents[1] / ".github" / "scripts" / "analyze_flaky_tests.py"
)
_SPEC = importlib.util.spec_from_file_location("analyze_flaky_tests", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("connection reset"),
        SimpleNamespace(status_code=503, content=b""),
    ],
)
def test_download_artifact_recovers_from_transient_failure(failure):
    success = SimpleNamespace(status_code=200, content=b"artifact")
    with (
        patch.object(analyze.requests, "get", side_effect=[failure, success]),
        patch.object(analyze.time, "sleep"),
    ):
        assert (
            analyze.download_artifact("https://example.test/artifact", "token")
            == b"artifact"
        )


def test_download_artifact_honors_retry_after_header():
    rate_limited = SimpleNamespace(
        status_code=429,
        headers={"Retry-After": "120"},
        content=b"",
    )
    success = SimpleNamespace(status_code=200, content=b"artifact")
    sleeps: list[float] = []
    with (
        patch.object(analyze.requests, "get", side_effect=[rate_limited, success]),
        patch.object(analyze.time, "sleep", side_effect=sleeps.append),
    ):
        assert (
            analyze.download_artifact("https://example.test/artifact", "token")
            == b"artifact"
        )

    assert sleeps == [120]


def test_download_artifact_preserves_persistent_failure():
    with (
        patch.object(
            analyze.requests,
            "get",
            side_effect=requests.ConnectionError("persistent failure"),
        ),
        patch.object(analyze.time, "sleep"),
        pytest.raises(requests.ConnectionError, match="persistent failure"),
    ):
        analyze.download_artifact("https://example.test/artifact", "token")


if __name__ == "__main__":
    pytest.main([__file__])
