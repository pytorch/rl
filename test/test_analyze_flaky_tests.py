from __future__ import annotations

import importlib.util
from pathlib import Path

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


class _Response:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


@pytest.mark.parametrize(
    "failure", [requests.ConnectionError("connection reset"), _Response(503)]
)
def test_download_artifact_retries_transient_failures(monkeypatch, failure):
    responses = iter([failure, _Response(200, b"artifact")])
    sleeps: list[int] = []

    def get(*args, **kwargs):
        response = next(responses)
        if isinstance(response, requests.RequestException):
            raise response
        return response

    monkeypatch.setattr(analyze.requests, "get", get)
    monkeypatch.setattr(analyze.time, "sleep", sleeps.append)

    assert (
        analyze.download_artifact("https://example.test/artifact", "token")
        == b"artifact"
    )
    assert sleeps == [2]


def test_download_artifact_reraises_exhausted_request_errors(monkeypatch):
    errors = iter(
        requests.ConnectionError("persistent failure")
        for _ in range(analyze.ARTIFACT_DOWNLOAD_ATTEMPTS)
    )
    sleeps: list[int] = []

    def get(*args, **kwargs):
        raise next(errors)

    monkeypatch.setattr(analyze.requests, "get", get)
    monkeypatch.setattr(analyze.time, "sleep", sleeps.append)

    with pytest.raises(requests.ConnectionError, match="persistent failure"):
        analyze.download_artifact("https://example.test/artifact", "token")
    assert sleeps == [2, 4]


def test_download_artifact_does_not_retry_permanent_http_status(monkeypatch):
    calls = 0

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(404)

    monkeypatch.setattr(analyze.requests, "get", get)
    monkeypatch.setattr(analyze.time, "sleep", pytest.fail)

    assert analyze.download_artifact("https://example.test/artifact", "token") is None
    assert calls == 1


if __name__ == "__main__":
    pytest.main([__file__])
