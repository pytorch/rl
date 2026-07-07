# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from multiprocessing.connection import wait as wait_for_connection

from typing import Any, TYPE_CHECKING, TypeVar

from torchrl._comm import Mailbox
from torchrl.record.loggers._service import (
    _flush_logger,
    _LoggerClient,
    _shutdown_logger,
)
from torchrl.record.loggers.common import Logger

__all__ = ["ProcessLogger"]

LoggerT = TypeVar("LoggerT", bound=Logger)

if TYPE_CHECKING:
    from typing import Self


def _unused_response_queue():
    raise RuntimeError("The logger service cannot create caller response queues.")


def _watch_logger_process(process_sentinel, alive_event) -> None:
    """Clear the shared liveness flag when the logger process exits."""
    try:
        wait_for_connection([process_sentinel])
    finally:
        try:
            alive_event.clear()
        except Exception:
            pass


def _logger_process_entry(
    logger_cls,
    args: tuple,
    kwargs: dict[str, Any],
    request_queue,
    response_queues,
    ready_queue,
    alive_event,
) -> None:
    logger = None
    try:
        logger = logger_cls(*args, **kwargs)
        alive_event.set()
        ready_queue.put(
            (
                True,
                {
                    "exp_name": getattr(logger, "exp_name", None),
                    "log_dir": getattr(logger, "log_dir", None),
                },
            )
        )
    except BaseException as error:
        ready_queue.put((False, repr(error)))
        raise

    mailbox = Mailbox(
        request_queue,
        _unused_response_queue,
        response_queues=response_queues,
    )
    errors: list[str] = []
    running = True
    try:
        while running:
            mailbox.wait_for_work(timeout=0.1)
            payloads, callbacks, _ = mailbox.drain(64)
            for payload, callback in zip(payloads, callbacks):
                method = payload["method"]
                wait = payload["wait"]
                try:
                    if method == "__flush__":
                        _flush_logger(logger)
                        if errors:
                            error_text = errors.pop(0)
                            raise RuntimeError(error_text)
                        result = None
                    elif method == "__shutdown__":
                        _shutdown_logger(logger)
                        result = None
                        running = False
                    elif method == "__repr__":
                        result = repr(logger)
                    else:
                        result = getattr(logger, method)(
                            *payload["args"], **payload["kwargs"]
                        )
                except BaseException as error:
                    remote_error = RuntimeError(
                        f"Logger service command {method!r} failed: {error!r}"
                    )
                    if wait:
                        mailbox.reject(callback, remote_error)
                    else:
                        errors.append(str(remote_error))
                else:
                    if wait:
                        mailbox.resolve(callback, result)
    finally:
        alive_event.clear()
        if running and logger is not None:
            try:
                _shutdown_logger(logger)
            except Exception:
                pass


class _ProcessLoggerClient(_LoggerClient):
    def __init__(self, mailbox_client, *, exp_name, log_dir) -> None:
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        self._mailbox_client = mailbox_client

    def _submit(
        self,
        method: str,
        args: tuple,
        kwargs: dict[str, Any],
        *,
        wait: bool,
        timeout: float | None = None,
    ) -> Any:
        future = self._mailbox_client.submit(
            {"method": method, "args": args, "kwargs": kwargs, "wait": wait}
        )
        if wait:
            return future.result(timeout=timeout)
        return None


class ProcessLogger(_ProcessLoggerClient):
    """Driver-owned logger service running in a dedicated process.

    The concrete logger is constructed once in the child process. Worker-side
    clients can only submit ``log_*`` calls; only this owner can flush or stop
    the service.

    Args:
        logger_cls: Concrete :class:`~torchrl.record.loggers.Logger` class.
        *args: Positional arguments passed to ``logger_cls``.
        mp_context: Multiprocessing context or start-method name. Defaults to
            ``"spawn"``.
        max_queue_size: Maximum number of pending logging commands. Defaults
            to ``1000``.
        startup_timeout: Seconds to wait for logger construction. Defaults to
            ``60``.
        **kwargs: Keyword arguments passed to ``logger_cls``.

    Examples:
        >>> from torchrl.record.loggers import CSVLogger, ProcessLogger
        >>> logger = ProcessLogger(CSVLogger, exp_name="run", log_dir="/tmp")
        >>> worker_logger = logger.client()
        >>> worker_logger.log_scalar("loss", 1.0, step=0)
        >>> logger.shutdown()
    """

    def __init__(
        self,
        logger_cls: type[LoggerT],
        *args: Any,
        mp_context: str | mp.context.BaseContext | None = None,
        max_queue_size: int = 1000,
        startup_timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        if isinstance(mp_context, str):
            self._ctx = mp.get_context(mp_context)
        elif mp_context is None:
            self._ctx = mp.get_context("spawn")
        else:
            self._ctx = mp_context
        self._service_cls = logger_cls
        self._logger_args = args
        self._logger_kwargs = kwargs
        self._startup_timeout = startup_timeout
        self._manager = self._ctx.Manager()
        self._request_queue = self._manager.Queue(maxsize=max_queue_size)
        self._response_queues = self._manager.dict()
        self._service_alive = self._manager.Event()
        self._mailbox = Mailbox(
            self._request_queue,
            self._manager.Queue,
            response_queues=self._response_queues,
            peer_alive=self._service_alive,
        )
        self._ready_queue = self._ctx.Queue()
        self._process: mp.Process | None = None
        self._process_monitor: threading.Thread | None = None
        self._closed = False
        self.start()
        try:
            ok, payload = self._ready_queue.get(timeout=startup_timeout)
        except queue.Empty:
            self.shutdown(timeout=1.0)
            raise TimeoutError(
                f"ProcessLogger did not start within {startup_timeout} seconds."
            ) from None
        if not ok:
            self.shutdown(timeout=1.0)
            raise RuntimeError(f"ProcessLogger failed to start: {payload}")
        self._metadata = payload
        owner_client = self._make_client()
        super().__init__(
            owner_client._mailbox_client,
            exp_name=payload["exp_name"],
            log_dir=payload["log_dir"],
        )

    def _make_client(self) -> _ProcessLoggerClient:
        metadata = getattr(self, "_metadata", {"exp_name": None, "log_dir": None})
        return _ProcessLoggerClient(
            self._mailbox.client(),
            exp_name=metadata["exp_name"],
            log_dir=metadata["log_dir"],
        )

    def start(self) -> Self:
        """Start the logger process and return this owner."""
        if self._closed:
            raise RuntimeError("A closed ProcessLogger cannot be restarted.")
        if self.is_alive:
            return self
        self._service_alive.clear()
        self._process = self._ctx.Process(
            target=_logger_process_entry,
            args=(
                self._service_cls,
                self._logger_args,
                self._logger_kwargs,
                self._request_queue,
                self._response_queues,
                self._ready_queue,
                self._service_alive,
            ),
            name="ProcessLogger",
        )
        self._process.start()
        self._process_monitor = threading.Thread(
            target=_watch_logger_process,
            args=(self._process.sentinel, self._service_alive),
            daemon=True,
            name="ProcessLoggerMonitor",
        )
        self._process_monitor.start()
        return self

    @property
    def is_alive(self) -> bool:
        """Whether the logger process is alive."""
        return self._process is not None and self._process.is_alive()

    def client(self) -> _ProcessLoggerClient:
        """Return a picklable logger client without lifecycle methods."""
        if not self.is_alive:
            raise RuntimeError("ProcessLogger is not running.")
        return self._make_client()

    @property
    def service_backend(self) -> str:
        """The canonical deployment backend for this logger."""
        return "process"

    def flush(self, timeout: float | None = None) -> None:
        """Wait for prior commands and propagate service-side failures."""
        self._submit("__flush__", (), {}, wait=True, timeout=timeout)

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Flush, stop the logger process, and release queue resources."""
        if self._closed:
            return
        process = self._process
        monitor = self._process_monitor
        error: BaseException | None = None

        def capture(caught: BaseException) -> None:
            nonlocal error
            if error is None:
                error = caught

        if process is not None:
            if process.is_alive():
                try:
                    self.flush(timeout=timeout)
                except BaseException as caught:
                    capture(caught)
                try:
                    self._submit("__shutdown__", (), {}, wait=True, timeout=timeout)
                except BaseException as caught:
                    capture(caught)
            try:
                process.join(timeout=timeout)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=timeout)
            except BaseException as caught:
                capture(caught)
            if monitor is not None:
                try:
                    monitor.join(timeout=timeout)
                except BaseException as caught:
                    capture(caught)
            try:
                process.close()
            except BaseException as caught:
                capture(caught)
        self._process = None
        self._process_monitor = None
        self._closed = True
        try:
            self._ready_queue.close()
        except BaseException as caught:
            capture(caught)
        try:
            self._ready_queue.join_thread()
        except BaseException as caught:
            capture(caught)
        try:
            self._manager.shutdown()
        except BaseException as caught:
            capture(caught)
        if error is not None:
            raise error

    def close(self, timeout: float | None = 5.0) -> None:
        """Alias for :meth:`shutdown`."""
        self.shutdown(timeout=timeout)
