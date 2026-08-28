from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CHECKER = _ROOT / "scripts" / "check-docstring-args"


def _run_checker(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(_CHECKER), str(path)],
        cwd=_ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_args_section_documents_positional_args(tmp_path: Path):
    result = _run_checker(
        tmp_path,
        '''
def helper(first, second):
    """Do work.

    Args:
        first: The first value.
        second: The second value.
    """
''',
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


def test_keyword_args_section_documents_keyword_only_args(tmp_path: Path):
    result = _run_checker(
        tmp_path,
        '''
def register_coeff_buffer(name, value, *, persistent=True, target=True):
    """Register a coefficient buffer.

    Args:
        name: The buffer name.
        value: The buffer value.

    Keyword Args:
        persistent: Whether the buffer is persistent.
        target: Whether the buffer is mirrored on the target module.
    """
''',
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


def test_mixed_args_and_keyword_args_sections(tmp_path: Path):
    result = _run_checker(
        tmp_path,
        '''
def helper(first, second, *, mode="strict", retries=0):
    """Do work.

    Args:
        first: The first value.
        second: The second value.

    Keyword Args:
        mode: The execution mode.
        retries: The retry count.
    """
''',
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


def test_missing_args_still_fail(tmp_path: Path):
    result = _run_checker(
        tmp_path,
        '''
def helper(first, second, *, mode="strict", retries=0):
    """Do work.

    Args:
        first: The first value.

    Keyword Args:
        mode: The execution mode.
    """
''',
    )

    assert result.returncode == 1
    assert "sample.py:2: helper" in result.stdout
    assert "second" in result.stdout
    assert "retries" in result.stdout
    assert "mode" not in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
