# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import threading
import warnings

from typing import Any, TYPE_CHECKING, TypeVar

from torchrl.record.loggers._service import (
    _flush_logger,
    _LoggerClient,
    _shutdown_logger,
)
from torchrl.record.loggers.common import Logger

__all__ = ["RayLogger"]

_has_ray = importlib.util.find_spec("ray") is not None
LoggerT = TypeVar("LoggerT", bound=Logger)

if TYPE_CHECKING:
    from typing import Self


def _make_remote_wrapper(logger_cls):
    class _RemoteWrapper(logger_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._service_errors: list[str] = []
            self._next_client_id = 0
            self._next_sequence: dict[int, int] = {}

        def _new_client(self) -> int:
            client_id = self._next_client_id
            self._next_client_id += 1
            self._next_sequence[client_id] = 0
            return client_id

        def _execute(
            self,
            client_id: int,
            sequence: int,
            method: str,
            args: tuple,
            kwargs: dict[str, Any],
            wait: bool,
        ):
            expected = self._next_sequence[client_id]
            if sequence != expected:
                error = RuntimeError(
                    f"Out-of-order logger command for client {client_id}: "
                    f"expected {expected}, got {sequence}."
                )
                if wait:
                    raise error
                self._service_errors.append(str(error))
                return None
            self._next_sequence[client_id] += 1
            try:
                if method == "__repr__":
                    return repr(self)
                return getattr(self, method)(*args, **kwargs)
            except BaseException as error:
                remote_error = RuntimeError(
                    f"Logger service command {method!r} failed: {error!r}"
                )
                if wait:
                    raise remote_error from error
                self._service_errors.append(str(remote_error))
                return None

        def _flush_service(self) -> None:
            _flush_logger(self)
            if self._service_errors:
                raise RuntimeError(self._service_errors.pop(0))

        def _shutdown_service(self) -> None:
            _shutdown_logger(self)

        def _metadata(self) -> dict[str, Any]:
            return {
                "exp_name": getattr(self, "exp_name", None),
                "log_dir": getattr(self, "log_dir", None),
            }

    _RemoteWrapper.__name__ = f"_Remote{logger_cls.__name__}"
    _RemoteWrapper.__qualname__ = f"_Remote{logger_cls.__name__}"
    return _RemoteWrapper


class _RayLoggerClient(_LoggerClient):
    def __init__(self, actor, client_id: int, *, exp_name, log_dir) -> None:
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        self._actor = actor
        self._client_id = client_id
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._ray = None

    @property
    def ray(self):
        if self._ray is None:
            import ray

            self._ray = ray
        return self._ray

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_ray"] = None
        state["_sequence_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._sequence_lock = threading.Lock()

    def _submit(
        self,
        method: str,
        args: tuple,
        kwargs: dict[str, Any],
        *,
        wait: bool,
        timeout: float | None = None,
    ) -> Any:
        with self._sequence_lock:
            sequence = self._sequence
            self._sequence += 1
            result = self._actor._execute.remote(
                self._client_id, sequence, method, args, kwargs, wait
            )
        if wait:
            return self.ray.get(result, timeout=timeout)
        return None


class RayLogger(_RayLoggerClient):
    """Driver-owned Ray logger service with restricted worker clients.

    Existing direct construction and ``use_ray_service=True`` continue to
    create this owner. Use :meth:`client` before sending the logger to workers.

    Args:
        logger_cls: Concrete :class:`~torchrl.record.loggers.Logger` class.
        *args: Positional arguments forwarded to ``logger_cls``.
        ray_actor_options: Options used to construct the Ray actor.
        ray_init_config: Options used to initialize Ray when needed.
        **kwargs: Keyword arguments forwarded to ``logger_cls``.

    Examples:
        >>> from torchrl.record import CSVLogger, RayLogger
        >>> logger = RayLogger(CSVLogger, exp_name="run", log_dir="/tmp")  # doctest: +SKIP
        >>> client = logger.client()  # doctest: +SKIP
        >>> client.log_scalar("loss", 1.0, step=0)  # doctest: +SKIP
        >>> logger.shutdown()  # doctest: +SKIP
    """

    def __init__(
        self,
        logger_cls: type[LoggerT],
        *args: Any,
        ray_actor_options: dict[str, Any] | None = None,
        ray_init_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not _has_ray:
            raise ImportError(
                "Ray is required for RayLogger. Install with: pip install ray"
            )
        import ray

        if not ray.is_initialized():
            ray.init(**(ray_init_config or {}))
        self._service_cls = logger_cls
        wrapper_cls = _make_remote_wrapper(logger_cls)
        actor_options = dict(ray_actor_options or {})
        actor_options.setdefault("max_pending_calls", 1000)
        remote_cls = ray.remote(**actor_options)(wrapper_cls)
        self._actor = remote_cls.remote(*args, **kwargs)
        metadata = ray.get(self._actor._metadata.remote())
        client_id = ray.get(self._actor._new_client.remote())
        self._closed = False
        super().__init__(
            self._actor,
            client_id,
            exp_name=metadata["exp_name"],
            log_dir=metadata["log_dir"],
        )
        self._ray = ray

    def start(self) -> Self:
        """Return this already-started Ray service owner."""
        if not self.is_alive:
            raise RuntimeError("A stopped RayLogger cannot be restarted.")
        return self

    @property
    def is_alive(self) -> bool:
        """Whether the Ray actor is available."""
        return not self._closed and self._actor is not None

    def client(self) -> _RayLoggerClient:
        """Return a Ray logger client without lifecycle methods."""
        if not self.is_alive:
            raise RuntimeError("RayLogger is not running.")
        client_id = self._ray.get(self._actor._new_client.remote())
        return _RayLoggerClient(
            self._actor,
            client_id,
            exp_name=self.exp_name,
            log_dir=self.log_dir,
        )

    @property
    def service_backend(self) -> str:
        """The canonical deployment backend for this logger."""
        return "ray"

    def flush(self, timeout: float | None = None) -> None:
        """Wait for queued actor calls and propagate logging failures."""
        self._ray.get(self._actor._flush_service.remote(), timeout=timeout)

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Flush and terminate the owned Ray actor."""
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self.flush(timeout=timeout)
        except BaseException as caught:
            error = caught
        try:
            self._ray.get(self._actor._shutdown_service.remote(), timeout=timeout)
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            self._ray.kill(self._actor, no_restart=True)
        finally:
            self._actor = None
            self._closed = True
        if error is not None:
            raise error

    def close(self, timeout: float | None = 5.0) -> None:
        """Alias for :meth:`shutdown`."""
        self.shutdown(timeout=timeout)

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        warnings.warn(
            "Implicit RayLogger shutdown from __del__ is deprecated and will "
            "stop terminating the actor in v0.16. Call shutdown() explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            self.shutdown(timeout=1.0)
        except Exception:
            pass
