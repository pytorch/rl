# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest

from torchrl.benchmark.cli import main as cli_main


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["torchrl-benchmark-rnn", *argv])
    cli_main()


def test_benchmark_cli_rejects_invalid_backend(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["--backends", "cudnn,bogus", "--device", "cpu"])


def test_benchmark_cli_rejects_invalid_compile_mode(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["--compile-modes", "none,fast", "--device", "cpu"])


def test_benchmark_cli_rejects_block_gru_recompute(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(
            monkeypatch,
            ["--rnn", "block_gru", "--recompute", "full", "--device", "cpu"],
        )


def test_benchmark_cli_skips_without_cuda(monkeypatch, capsys):
    _run_cli(monkeypatch, ["--device", "cpu"])
    assert "CUDA required" in capsys.readouterr().out


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
