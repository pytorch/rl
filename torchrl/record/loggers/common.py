# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDictBase
from torch import Tensor


__all__ = ["Logger"]


def _make_metrics_safe(
    metrics: dict[str, Any] | TensorDictBase,
    *,
    keys_sep: str = "/",
) -> dict[str, Any]:
    """Convert metric values to be safe for cross-process logging.

    This function converts torch tensors to CPU/Python types, which is
    necessary when logging metrics to external services (e.g., wandb, mlflow)
    that may run in separate processes without GPU access.

    For regular dicts, the implementation batches CUDA->CPU transfers using
    non_blocking=True and synchronizes once via a CUDA event, avoiding the
    overhead of multiple implicit synchronizations that would occur if calling
    .item() on each CUDA tensor individually.

    For TensorDict inputs, this leverages TensorDict's efficient batch `.to()`
    method which transfers all tensors in a single operation.

    Args:
        metrics: Dictionary or TensorDict of metric names to values. Values can
            be torch.Tensor (CUDA or CPU), Python scalars, or other types.
        keys_sep: Separator used to flatten nested TensorDict keys into strings.
            Defaults to "/". Only used for TensorDict inputs.

    Returns:
        Dictionary with the same keys but tensor values converted to
        Python scalars (for single-element tensors) or lists (for
        multi-element tensors). Non-tensor values are passed through unchanged.
    """
    if isinstance(metrics, TensorDictBase):
        return _make_metrics_safe_tensordict(metrics, keys_sep=keys_sep)

    out: dict[str, Any] = {}
    cpu_tensors: dict[str, Tensor] = {}
    has_cuda_tensors = False

    # First pass: identify tensors and start non-blocking CUDA->CPU transfers
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            if value.is_cuda:
                # Non-blocking transfer - queues the copy without waiting
                value = value.detach().to("cpu", non_blocking=True)
                has_cuda_tensors = True
            else:
                value = value.detach()
            cpu_tensors[key] = value
        else:
            out[key] = value

    # Explicit sync: use a CUDA event instead of global synchronize() - this
    # only waits for work up to the point the event was recorded, not ALL
    # pending GPU work.
    if has_cuda_tensors:
        event = torch.cuda.Event()
        event.record()
        event.synchronize()

    # Second pass: convert CPU tensors to Python scalars/lists
    for key, value in cpu_tensors.items():
        if value.numel() == 1:
            out[key] = value.item()
        else:
            out[key] = value.tolist()

    return out


def _make_metrics_safe_tensordict(
    metrics: TensorDictBase,
    *,
    keys_sep: str = "/",
) -> dict[str, Any]:
    """Convert TensorDict metric values to be safe for cross-process logging.

    This leverages TensorDict's efficient batch `.to()` method which transfers
    all tensors in a single operation, then converts to Python scalars.

    Args:
        metrics: TensorDict of metric names to tensor values.
        keys_sep: Separator used to flatten nested keys into strings.

    Returns:
        Dictionary with flattened string keys and Python scalar/list values.
    """
    # TensorDict's .to() efficiently batches all tensor transfers
    if metrics.device is not None and metrics.device.type == "cuda":
        metrics = metrics.to("cpu", non_blocking=True)
        # Sync after batched transfer
        event = torch.cuda.Event()
        event.record()
        event.synchronize()

    # Flatten nested keys and convert to dict
    flat_dict = metrics.flatten_keys(keys_sep).to_dict()

    # Convert tensors to Python scalars/lists
    out: dict[str, Any] = {}
    for key, value in flat_dict.items():
        if isinstance(value, Tensor):
            value = value.detach()
            if value.numel() == 1:
                out[key] = value.item()
            else:
                out[key] = value.tolist()
        else:
            out[key] = value

    return out


class Logger:
    """A template for loggers."""

    def __init__(self, exp_name: str, log_dir: str) -> None:
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.experiment = self._create_experiment()

    @abc.abstractmethod
    def _create_experiment(self) -> Experiment:  # noqa: F821
        ...

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        ...

    @abc.abstractmethod
    def log_video(
        self, name: str, video: Tensor, step: int | None = None, **kwargs
    ) -> None:
        ...

    @abc.abstractmethod
    def log_hparams(self, cfg: DictConfig | dict) -> None:  # noqa: F821
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        ...

    @abc.abstractmethod
    def log_histogram(self, name: str, data: Sequence, **kwargs):
        ...

    def log_metrics(
        self,
        metrics: dict[str, Any] | TensorDictBase,
        step: int | None = None,
        *,
        keys_sep: str = "/",
    ) -> dict[str, Any]:
        """Log multiple scalar metrics at once.

        This method efficiently handles tensor values by batching CUDA->CPU
        transfers and performing a single synchronization, avoiding the overhead
        of multiple implicit syncs that would occur when logging tensors one at
        a time.

        This is particularly useful when logging to services running in separate
        processes (e.g., Ray actors) that may not have GPU access.

        Args:
            metrics: Dictionary or TensorDict mapping metric names to values.
                Tensor values are automatically converted to Python scalars/lists.
                For TensorDict inputs, nested keys are flattened using ``keys_sep``.
            step: Optional step value for all metrics.
            keys_sep: Separator used to flatten nested TensorDict keys into strings.
                Defaults to "/". Only used for TensorDict inputs.

        Returns:
            The converted metrics dictionary (with tensors converted to Python types).
        """
        safe_metrics = _make_metrics_safe(metrics, keys_sep=keys_sep)
        for name, value in safe_metrics.items():
            self.log_scalar(name, value, step=step)
        return safe_metrics
