# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

__all__ = ["RayLogger"]


def _make_remote_wrapper(logger_cls):
    """Create a wrapper subclass with helper methods accessible via Ray remote calls.

    Ray actors can only call regular methods, not dunder methods like ``__repr__``
    or ``__getattribute__``. This wrapper adds ``_get_attr`` and ``_repr`` methods
    that are accessible via ``.remote()``.
    """

    class _RemoteWrapper(logger_cls):
        def _get_attr(self, name):
            return getattr(self, name)

        def _repr(self):
            return repr(self)

    _RemoteWrapper.__name__ = f"_Remote{logger_cls.__name__}"
    _RemoteWrapper.__qualname__ = f"_Remote{logger_cls.__name__}"
    return _RemoteWrapper


class RayLogger:
    """A generic Ray actor wrapper for any TorchRL Logger.

    This class wraps a logger as a Ray actor, delegating all method calls
    to the remote actor via ``ray.get(actor.method.remote(...))``.

    CUDA tensors in :meth:`log_metrics` and :meth:`log_video` are automatically
    moved to CPU before the remote call.

    This class is not meant to be instantiated directly. Instead, pass
    ``use_ray_service=True`` when constructing any Logger subclass::

        >>> logger = WandbLogger(exp_name="test", use_ray_service=True)
        >>> logger = CSVLogger(exp_name="test", log_dir="/tmp", use_ray_service=True,
        ...                    ray_actor_options={"num_cpus": 1})

    Args:
        logger_cls: The logger class to wrap as a Ray actor.
        *args: Positional arguments passed to the logger constructor.

    Keyword Args:
        ray_actor_options (dict, optional): Options passed to ``ray.remote()``
            when creating the Ray actor (e.g., ``{"num_cpus": 1, "num_gpus": 0}``).
        **kwargs: Keyword arguments passed to the logger constructor.
    """

    def __init__(self, logger_cls, *args, ray_actor_options=None, **kwargs):
        try:
            import ray
        except ImportError:
            raise ImportError(
                "Ray is required for RayLogger. Install with: pip install ray"
            )
        self._ray = ray
        self._logger_cls = logger_cls

        wrapper_cls = _make_remote_wrapper(logger_cls)
        if ray_actor_options:
            RemoteLoggerCls = ray.remote(**ray_actor_options)(wrapper_cls)
        else:
            RemoteLoggerCls = ray.remote(wrapper_cls)
        self._actor = RemoteLoggerCls.remote(*args, **kwargs)

    # --- Core Logger methods ---

    def log_scalar(
        self, name: str, value: float, step: int | None = None, **kwargs
    ) -> None:
        """Log a scalar value. See :meth:`~torchrl.record.loggers.Logger.log_scalar`."""
        self._ray.get(self._actor.log_scalar.remote(name, value, step=step, **kwargs))

    def log_video(
        self, name: str, video: Tensor, step: int | None = None, **kwargs
    ) -> None:
        """Log a video tensor. See :meth:`~torchrl.record.loggers.Logger.log_video`.

        The video tensor is moved to CPU before sending to the remote actor.
        """
        if hasattr(video, "cpu"):
            video = video.cpu()
        self._ray.get(self._actor.log_video.remote(name, video, step=step, **kwargs))

    def log_hparams(self, cfg) -> None:
        """Log hyperparameters. See :meth:`~torchrl.record.loggers.Logger.log_hparams`."""
        self._ray.get(self._actor.log_hparams.remote(cfg))

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        """Log histogram data. See :meth:`~torchrl.record.loggers.Logger.log_histogram`."""
        self._ray.get(self._actor.log_histogram.remote(name, data, **kwargs))

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        *,
        keys_sep: str = "/",
    ) -> dict[str, Any]:
        """Log multiple scalar metrics at once.

        See :meth:`~torchrl.record.loggers.Logger.log_metrics`.

        CUDA tensors are converted to Python scalars before the remote call.
        """
        from torchrl.record.loggers.common import _make_metrics_safe

        safe_metrics = _make_metrics_safe(metrics, keys_sep=keys_sep)
        self._ray.get(
            self._actor.log_metrics.remote(safe_metrics, step=step, keys_sep=keys_sep)
        )
        return safe_metrics

    def __repr__(self) -> str:
        return self._ray.get(self._actor._repr.remote())

    # --- Properties delegated to remote actor ---

    @property
    def exp_name(self):
        """The experiment name."""
        return self._ray.get(self._actor._get_attr.remote("exp_name"))

    @property
    def log_dir(self):
        """The log directory."""
        return self._ray.get(self._actor._get_attr.remote("log_dir"))

    # --- Generic fallback for logger-specific methods ---

    def __getattr__(self, name):
        # Guard against private/dunder attributes to prevent infinite recursion
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        actor = self.__dict__.get("_actor")
        ray = self.__dict__.get("_ray")
        if actor is None or ray is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def _remote_call(*args, **kwargs):
            return ray.get(getattr(actor, name).remote(*args, **kwargs))

        return _remote_call

    def __del__(self):
        if hasattr(self, "_ray") and hasattr(self, "_actor"):
            try:
                self._ray.kill(self._actor, no_restart=True)
            except Exception:
                pass
