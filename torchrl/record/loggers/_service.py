# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tensordict import TensorDictBase
from torch import Tensor

from torchrl.record.loggers.common import _make_metrics_safe


class _LoggerClient:
    """Domain-compatible logger client without lifecycle capabilities."""

    def __init__(self, *, exp_name: str | None, log_dir: str | None) -> None:
        self._exp_name = exp_name
        self._log_dir = log_dir

    def _submit(
        self,
        method: str,
        args: tuple,
        kwargs: dict[str, Any],
        *,
        wait: bool,
        timeout: float | None = None,
    ) -> Any:
        raise NotImplementedError

    def log_scalar(
        self, name: str, value: float, step: int | None = None, **kwargs
    ) -> None:
        """Log a scalar and wait for service-side completion."""
        self._submit("log_scalar", (name, value), {"step": step, **kwargs}, wait=True)

    def log_video(
        self, name: str, video: Tensor, step: int | None = None, **kwargs
    ) -> None:
        """Log a CPU video and wait for encoding/upload acknowledgement."""
        if hasattr(video, "cpu"):
            video = video.cpu()
        self._submit("log_video", (name, video), {"step": step, **kwargs}, wait=True)

    def log_hparams(self, cfg) -> None:
        """Log hyperparameters and wait for service-side completion."""
        self._submit("log_hparams", (cfg,), {}, wait=True)

    def log_histogram(self, name: str, data: Sequence, **kwargs) -> None:
        """Log histogram data and wait for service-side completion."""
        self._submit("log_histogram", (name, data), kwargs, wait=True)

    def log_metrics(
        self,
        metrics: dict[str, Any] | TensorDictBase,
        step: int | None = None,
        *,
        keys_sep: str = "/",
        override_global_step: bool = False,
    ) -> dict[str, Any]:
        """Convert metrics locally and wait for service-side completion."""
        safe_metrics = _make_metrics_safe(metrics, keys_sep=keys_sep)
        kwargs = {"step": step, "keys_sep": keys_sep}
        if override_global_step:
            kwargs["override_global_step"] = True
        self._submit("log_metrics", (safe_metrics,), kwargs, wait=True)
        return safe_metrics

    @property
    def exp_name(self) -> str | None:
        """Experiment name reported by the owned logger."""
        return self._exp_name

    @property
    def log_dir(self) -> str | None:
        """Log directory reported by the owned logger."""
        return self._log_dir

    def __repr__(self) -> str:
        return self._submit("__repr__", (), {}, wait=True)

    def __getattr__(self, name: str):
        if not name.startswith("log_"):
            raise AttributeError(
                f"{type(self).__name__!s} has no lifecycle capability {name!r}."
            )

        def log_method(*args, **kwargs):
            return self._submit(name, args, kwargs, wait=True)

        return log_method


def _flush_logger(logger) -> None:
    flush = getattr(logger, "flush", None)
    if callable(flush):
        flush()
        return
    experiment = getattr(logger, "experiment", None)
    flush = getattr(experiment, "flush", None)
    if callable(flush):
        flush()


def _shutdown_logger(logger) -> None:
    _flush_logger(logger)
    experiment = getattr(logger, "experiment", None)
    finish = getattr(experiment, "finish", None)
    if callable(finish):
        finish()
        return
    close = getattr(experiment, "close", None)
    if callable(close):
        close()
