# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import importlib.util
from collections.abc import Sequence

from typing import Any, TYPE_CHECKING

import torch
from tensordict import TensorDictBase
from torch import Tensor

from torchrl._utils import _RayServiceMetaClass

_has_tv = importlib.util.find_spec("torchvision") is not None
_has_torchcodec = importlib.util.find_spec("torchcodec") is not None

if TYPE_CHECKING:
    from typing import Self


__all__ = ["Logger"]


def _write_video(filename, video_array, **kwargs):
    if not _has_torchcodec:
        raise ModuleNotFoundError(
            "Writing MP4 videos with VideoRecorder or CSVLogger requires "
            "torchcodec >= 0.10.0. When running TorchRL from this repository "
            "with uv, use `uv run --extra video <command>` (or "
            "`uv run --extra rendering <command>`) so torchcodec is installed "
            "in the command environment. Otherwise install it with "
            "`pip install 'torchcodec>=0.10.0'`."
        )
    try:
        from torchcodec.encoders import VideoEncoder
    except Exception as err:
        raise ImportError(
            "torchcodec is installed but could not be imported for MP4 video "
            "writing. Make sure the installed torchcodec build is compatible "
            "with the active PyTorch build, or rerun the command with "
            "`uv run --extra video <command>` from the TorchRL repository."
        ) from err

    fps = kwargs.pop("fps", 30)
    video_codec = kwargs.pop("video_codec", None)
    options = dict(kwargs.pop("options", None) or {})
    crf = options.pop("crf", None)
    preset = options.pop("preset", None)
    pixel_format = options.pop("pixel_format", None)

    # VideoEncoder expects (N, C, H, W); callers pass (T, H, W, C)
    video_array = video_array.permute(0, 3, 1, 2).contiguous()

    to_file_kwargs = {}
    if video_codec is not None:
        to_file_kwargs["codec"] = video_codec
    if crf is not None:
        to_file_kwargs["crf"] = float(crf)
    if preset is not None:
        to_file_kwargs["preset"] = preset
    if pixel_format is not None:
        to_file_kwargs["pixel_format"] = pixel_format
    if options:
        to_file_kwargs["extra_options"] = options

    VideoEncoder(frames=video_array, frame_rate=fps).to_file(filename, **to_file_kwargs)


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
    metrics = metrics.to("cpu", non_blocking=True)

    # Sync if CUDA is in use - the event sync is cheap if no work is pending
    if torch.cuda.is_initialized():
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


class Logger(metaclass=_RayServiceMetaClass):
    """A template for loggers.

    Keyword Args:
        service_backend (str): Deployment backend. One of ``"direct"``,
            ``"process"``, or ``"ray"``. Defaults to ``"direct"``.
        service_backend_options (dict, optional): Backend options. Process
            services accept ``context``/``mp_context``, ``max_queue_size``, and
            ``startup_timeout``. Ray services accept ``actor_options`` and
            ``ray_init_config``.
        use_ray_service (bool): If ``True``, the logger runs as a Ray actor
            in a separate process. Deprecated in favor of
            ``service_backend="ray"`` and scheduled for removal in v0.16.
            Defaults to ``False``.
        ray_actor_options (dict, optional): Options passed to ``ray.remote()``
            when creating the Ray actor (e.g., ``{"num_cpus": 1}``).
            Only used when ``use_ray_service=True``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _concrete_cls = cls

        def _ray_wrapper(*args, ray_actor_options=None, **kwargs):
            from torchrl.record.loggers.ray import RayLogger

            return RayLogger(
                _concrete_cls, *args, ray_actor_options=ray_actor_options, **kwargs
            )

        def _service_wrapper(
            service_backend,
            *args,
            service_backend_options=None,
            **kwargs,
        ):
            options = dict(service_backend_options or {})
            if service_backend == "process":
                from torchrl.record.loggers.process import ProcessLogger

                if "context" in options:
                    if "mp_context" in options:
                        raise ValueError(
                            "Use only one of 'context' and 'mp_context' in "
                            "service_backend_options."
                        )
                    options["mp_context"] = options.pop("context")
                return ProcessLogger(_concrete_cls, *args, **options, **kwargs)
            if service_backend == "ray":
                from torchrl.record.loggers.ray import RayLogger

                legacy_actor_options = kwargs.pop("ray_actor_options", None)
                actor_options = options.pop("actor_options", legacy_actor_options)
                ray_init_config = options.pop("ray_init_config", None)
                if options:
                    raise TypeError(
                        f"Unexpected Ray logger service options: {sorted(options)}"
                    )
                return RayLogger(
                    _concrete_cls,
                    *args,
                    ray_actor_options=actor_options,
                    ray_init_config=ray_init_config,
                    **kwargs,
                )
            raise ValueError(
                f"Logger does not support service_backend={service_backend!r}."
            )

        cls._RayServiceClass = staticmethod(_ray_wrapper)
        cls._ServiceClass = staticmethod(_service_wrapper)

    def __init__(self, exp_name: str, log_dir: str) -> None:
        self.exp_name = exp_name
        self.log_dir = log_dir
        self._service_shutdown = False
        self.experiment = self._create_experiment()

    def start(self) -> Self:
        """Return this already-started direct logger."""
        if self._service_shutdown:
            raise RuntimeError("A shut down direct logger cannot be restarted.")
        return self

    @property
    def is_alive(self) -> bool:
        """Whether the direct logger remains available."""
        return not self._service_shutdown

    def client(self) -> Self:
        """Return ``self`` for the zero-overhead direct backend."""
        return self

    @property
    def service_backend(self) -> str:
        """The canonical deployment backend for this logger."""
        return "direct"

    def flush(self, timeout: float | None = None) -> None:
        """Flush the underlying experiment when it exposes ``flush``."""
        del timeout
        flush = getattr(self.experiment, "flush", None)
        if callable(flush):
            flush()

    def shutdown(self, timeout: float | None = None) -> None:
        """Flush and close the underlying direct experiment."""
        del timeout
        if self._service_shutdown:
            return
        self.flush()
        finish = getattr(self.experiment, "finish", None)
        if callable(finish):
            finish()
        else:
            close = getattr(self.experiment, "close", None)
            if callable(close):
                close()
        self._service_shutdown = True

    def close(self, timeout: float | None = None) -> None:
        """Alias for :meth:`shutdown`."""
        self.shutdown(timeout=timeout)

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
