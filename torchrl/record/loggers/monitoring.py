# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from torchrl._utils import logger as torchrl_logger
from torchrl.record.loggers.common import Logger

__all__ = ["Every", "LoggerMonitor"]


_DEFAULT_RATE_NAMES = {
    "frames": "frames_per_second",
    "batches": "batches_per_second",
    "write_count": "writes_per_second",
    "worker_frames": "worker_frames_per_second",
    "sample_calls": "samples_per_second",
    "samples_returned": "samples_returned_per_second",
}


@dataclass(frozen=True)
class Every:
    """A logging schedule for :class:`~torchrl.record.loggers.monitoring.LoggerMonitor`.

    Schedules are built through the two factory methods rather than the
    constructor:

    - :meth:`Every.seconds` triggers on wall-clock time;
    - :meth:`Every.counter` triggers whenever a cumulative counter reported
      by the watched object's ``stats()`` snapshot crosses a multiple of
      ``interval``.

    Counter schedules are observed, not executed at the exact operation: if
    a counter jumps across several thresholds between two polls, the latest
    snapshot is logged once. A counter decrease (after a reset or a state
    restoration) re-baselines the schedule instead of producing spurious
    logs.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
        >>> from torchrl.record import CSVLogger
        >>> from torchrl.record.loggers.monitoring import Every, LoggerMonitor
        >>> logger = CSVLogger(exp_name="every_demo", log_dir=tempfile.mkdtemp())
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
        >>> monitor = LoggerMonitor(logger, background=False)
        >>> name = monitor.watch(
        ...     rb, name="rb", schedule=Every.counter("write_count", 50), log_on_start=False
        ... )
        >>> logged = monitor.step()  # primes the schedule, nothing logged yet
        >>> print(logged)
        {}
        >>> _ = rb.extend(torch.arange(60))
        >>> logged = monitor.step()  # write_count crossed the 50 threshold
        >>> print(logged["rb"]["rb/write_count"])
        60.0
    """

    kind: Literal["seconds", "counter"]
    interval: float
    key: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("seconds", "counter"):
            raise ValueError(f"kind must be 'seconds' or 'counter', got {self.kind!r}.")
        if not self.interval or self.interval <= 0:
            raise ValueError(
                f"interval must be strictly positive, got {self.interval}."
            )
        if self.kind == "counter" and not self.key:
            raise ValueError("counter schedules require a non-empty counter key.")

    @classmethod
    def seconds(cls, interval: float) -> Every:
        """Returns a wall-clock schedule triggering every ``interval`` seconds."""
        return cls(kind="seconds", interval=float(interval))

    @classmethod
    def counter(cls, key: str, interval: int) -> Every:
        """Returns a schedule triggering when the stats entry ``key`` crosses a multiple of ``interval``."""
        return cls(kind="counter", interval=interval, key=key)


class _Watch:
    """Internal bookkeeping for one monitored target."""

    def __init__(
        self,
        *,
        target: Any,
        name: str,
        schedule: Every | None,
        step_key: str | None,
        stats_fn: Callable[..., Mapping[str, Any]],
        stats_kwargs: dict[str, Any],
        rate_keys: tuple[str, ...] | None,
        log_on_start: bool,
        log_on_close: bool,
    ) -> None:
        self.target = target
        self.name = name
        self.schedule = schedule
        self.step_key = step_key
        self.stats_fn = stats_fn
        self.stats_kwargs = stats_kwargs
        self.rate_keys = rate_keys
        self.log_on_start = log_on_start
        self.log_on_close = log_on_close
        self.baseline_counters: dict[str, float] = {}
        self.baseline_time: float | None = None
        self.next_threshold: float | None = None
        self.last_counter_value: float | None = None
        self.last_log_time: float | None = None
        self.has_logged = False
        self.error_warned = False


class LoggerMonitor:
    """A pull-based monitor logging operational statistics of collectors and replay buffers.

    The monitor periodically requests a cheap ``stats()`` snapshot from each
    watched object, decides whether the snapshot is due for logging according
    to the per-target schedule, derives rates from cumulative counter deltas,
    namespaces the metrics as ``"<name>/<metric>"`` and forwards them to a
    single :class:`~torchrl.record.loggers.common.Logger`.

    Because snapshots are pulled from the monitor's own thread (or from
    explicit :meth:`step` calls), no logging work is ever executed on the
    collection, write or sampling hot paths of the watched objects, and a
    slow logging backend can only delay the next poll, never build an
    unbounded backlog. The monitor works with any object exposing a
    ``stats()`` method returning a flat mapping of scalars: in particular
    local, multiprocessing and Ray collectors and replay buffers share the
    same interface.

    The monitor never takes ownership of the logger or of the watched
    objects: stopping the monitor leaves both running, and shutting down the
    logger remains the caller's responsibility.

    Args:
        logger (Logger): the output sink. Any TorchRL logger works, including
            loggers running as a service (``service_backend="ray"``).

    Keyword Args:
        poll_interval (float, optional): how often the background thread
            polls the watched objects, in seconds. Counter schedules are
            evaluated at each poll, so this bounds their resolution.
            Defaults to ``1.0``.
        background (bool, optional): if ``True`` (default), entering the
            monitor context (or calling :meth:`start`) spawns a daemon
            thread that polls every ``poll_interval`` seconds. If ``False``,
            nothing runs in the background and the user drives the monitor
            through explicit :meth:`step` calls, which keeps tests and
            deterministic loops reproducible.
        on_error (str, optional): behavior when polling or logging a target
            fails: ``"warn"`` (default) logs a warning once per failure
            streak through the torchrl logger, ``"ignore"`` silently skips,
            and ``"raise"`` propagates the exception (in background mode
            this terminates the monitor thread).

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
        >>> from torchrl.record import CSVLogger
        >>> from torchrl.record.loggers.monitoring import Every, LoggerMonitor
        >>> logger = CSVLogger(exp_name="monitor_demo", log_dir=tempfile.mkdtemp())
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
        >>> monitor = LoggerMonitor(logger, background=False)
        >>> handle = monitor.watch(rb, name="replay_buffer", step="write_count")
        >>> _ = rb.extend(torch.arange(10))
        >>> logged = monitor.step()
        >>> print(logged["replay_buffer"]["replay_buffer/size"])
        10.0

    The typical background usage mirrors the RFC acceptance example:

        >>> with LoggerMonitor(logger) as monitor:  # doctest: +SKIP
        ...     monitor.watch(collector, name="collector", schedule=Every.counter("frames", 10_000))
        ...     monitor.watch(replay_buffer, name="replay_buffer", schedule=Every.seconds(5), step="write_count")
        ...     collector.start()
        ...     run_training()
    """

    def __init__(
        self,
        logger: Logger,
        *,
        poll_interval: float = 1.0,
        background: bool = True,
        on_error: Literal["warn", "raise", "ignore"] = "warn",
    ) -> None:
        if on_error not in ("warn", "raise", "ignore"):
            raise ValueError(
                f"on_error must be one of 'warn', 'raise' or 'ignore', got {on_error!r}."
            )
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {poll_interval}.")
        self._logger = logger
        self._poll_interval = float(poll_interval)
        self._background = background
        self._on_error = on_error
        self._watches: dict[str, _Watch] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False

    def watch(
        self,
        target: Any,
        *,
        name: str | None = None,
        schedule: Every | None = None,
        step: str | None = None,
        stats_fn: Callable[..., Mapping[str, Any]] | None = None,
        stats_kwargs: dict[str, Any] | None = None,
        rate_keys: tuple[str, ...] | None = None,
        log_on_start: bool = True,
        log_on_close: bool = True,
    ) -> str:
        """Registers an object to monitor and returns its watch name.

        Args:
            target: any object exposing a ``stats()`` method returning a flat
                mapping of scalars, such as collectors and replay buffers.

        Keyword Args:
            name (str, optional): namespace prefix for this target's metrics.
                Must be unique within the monitor. Defaults to the lowercase
                class name, uniquified with a numeric suffix if needed.
            schedule (Every, optional): when to log. ``None`` (default) logs
                at every poll.
            step (str, optional): name of a snapshot entry to use as the
                logical logging step (for example ``"write_count"``). For
                counter schedules this defaults to the schedule's counter
                key; otherwise the logger's own step handling applies.
            stats_fn (callable, optional): replaces ``target.stats`` as the
                snapshot provider. Runs in the monitor's process and thread.
            stats_kwargs (dict, optional): keyword arguments forwarded to the
                snapshot provider, for example ``{"workers": "both"}`` for
                distributed collectors. For multiprocessing collectors,
                per-worker snapshots travel over the collector control pipes
                and must not race with control calls (such as weight updates)
                issued from other threads; see
                :meth:`~torchrl.collectors.MultiSyncCollector.stats` for the
                thread-safety contract before enabling per-worker views on a
                background monitor.
            rate_keys (tuple of str, optional): snapshot entries to derive
                per-second rates from, based on their delta since the last
                logged snapshot. Defaults to the entries of the snapshot that
                are known cumulative counters (``frames``, ``batches``,
                ``write_count``, ...). A counter decrease re-baselines the
                rate instead of reporting a negative value.
            log_on_start (bool, optional): whether the first poll of this
                target logs unconditionally. When ``False``, the first poll
                only records baselines, so the first logged snapshot can
                carry rates. Defaults to ``True``.
            log_on_close (bool, optional): whether :meth:`stop` logs a final
                snapshot of this target. Defaults to ``True``.

        Returns:
            The watch name, usable with :meth:`unwatch`.
        """
        if stats_fn is None:
            stats_attr = getattr(target, "stats", None)
            if not callable(stats_attr):
                raise TypeError(
                    f"Cannot watch object of type {type(target).__name__}: it does not "
                    "expose a callable stats() method and no stats_fn was provided."
                )
            stats_fn = stats_attr
        with self._lock:
            if name is None:
                base = type(target).__name__.lower()
                name = base
                suffix = 0
                while name in self._watches:
                    suffix += 1
                    name = f"{base}_{suffix}"
            elif name in self._watches:
                raise ValueError(f"A watch named {name!r} is already registered.")
            if step is None and schedule is not None and schedule.kind == "counter":
                step = schedule.key
            self._watches[name] = _Watch(
                target=target,
                name=name,
                schedule=schedule,
                step_key=step,
                stats_fn=stats_fn,
                stats_kwargs=dict(stats_kwargs or {}),
                rate_keys=tuple(rate_keys) if rate_keys is not None else None,
                log_on_start=log_on_start,
                log_on_close=log_on_close,
            )
        return name

    def unwatch(self, name: str) -> None:
        """Stops monitoring the named target without touching the target itself."""
        with self._lock:
            if name not in self._watches:
                raise KeyError(f"No watch named {name!r}.")
            del self._watches[name]

    def step(self) -> dict[str, dict[str, float]]:
        """Polls every watched target once and logs the snapshots that are due.

        This is the manual-mode entry point; the background thread calls it
        on a timer. Returns a mapping from watch name to the logged payload,
        containing only the targets that were actually logged during this
        step.
        """
        logged: dict[str, dict[str, float]] = {}
        with self._lock:
            watches = list(self._watches.values())
        for watch in watches:
            payload = self._poll(watch)
            if payload is not None:
                logged[watch.name] = payload
        return logged

    def start(self) -> LoggerMonitor:
        """Starts the background polling thread (no-op when ``background=False``)."""
        self._stopped = False
        if self._background and self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="LoggerMonitor", daemon=True
            )
            self._thread.start()
        return self

    def stop(self, *, log_final: bool | None = None) -> None:
        """Stops polling and logs a final snapshot of each target.

        The watched objects and the logger are left untouched: shutting them
        down remains the caller's responsibility. Calling :meth:`stop` again
        without an intervening :meth:`start` is a no-op, so exiting the
        context manager after an explicit stop does not log the final
        snapshots twice.

        Keyword Args:
            log_final (bool, optional): overrides the per-watch
                ``log_on_close`` setting when provided.
        """
        self._stop_event.set()
        thread = self._thread
        polling_thread_stuck = False
        if thread is not None:
            thread.join(timeout=max(5.0, 2 * self._poll_interval))
            polling_thread_stuck = thread.is_alive()
            self._thread = None
        if self._stopped:
            return
        self._stopped = True
        if polling_thread_stuck:
            torchrl_logger.warning(
                "LoggerMonitor could not join its polling thread within the stop "
                "timeout (a stats() call is probably hanging); skipping the final "
                "snapshots to avoid polling concurrently with the stuck thread."
            )
            return
        with self._lock:
            watches = list(self._watches.values())
        for watch in watches:
            should_log = watch.log_on_close if log_final is None else log_final
            if should_log:
                self._poll(watch, force=True)

    def __enter__(self) -> LoggerMonitor:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            self.step()

    def _poll(self, watch: _Watch, force: bool = False) -> dict[str, float] | None:
        try:
            return self._poll_impl(watch, force=force)
        except Exception as err:
            if self._on_error == "raise":
                raise
            if self._on_error == "warn" and not watch.error_warned:
                watch.error_warned = True
                torchrl_logger.warning(
                    f"LoggerMonitor failed to poll or log watch {watch.name!r}: {err!r}. "
                    "Further errors for this watch will be silent until it recovers."
                )
            return None

    def _poll_impl(self, watch: _Watch, force: bool = False) -> dict[str, float] | None:
        raw = watch.stats_fn(**watch.stats_kwargs)
        snapshot = {
            key: float(value)
            for key, value in raw.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        now = time.monotonic()
        first_poll = watch.baseline_time is None
        due = force or self._is_due(watch, snapshot, now, first_poll=first_poll)
        payload = None
        if due:
            rates = self._compute_rates(watch, snapshot, now)
            payload = {
                f"{watch.name}/{key}": value
                for key, value in {**snapshot, **rates}.items()
            }
            step_value = None
            if watch.step_key is not None and watch.step_key in snapshot:
                step_value = int(snapshot[watch.step_key])
            self._logger.log_metrics(payload, step=step_value)
            watch.baseline_counters = dict(snapshot)
            watch.baseline_time = now
            watch.last_log_time = now
            watch.has_logged = True
        elif first_poll:
            watch.baseline_counters = dict(snapshot)
            watch.baseline_time = now
        watch.error_warned = False
        return payload

    def _is_due(
        self, watch: _Watch, snapshot: dict[str, float], now: float, *, first_poll: bool
    ) -> bool:
        schedule = watch.schedule
        if first_poll and not watch.has_logged:
            if watch.log_on_start:
                self._prime_schedule(watch, snapshot)
                return True
            if schedule is None:
                # log_on_start=False: the first poll only records baselines.
                return False
        if schedule is None:
            return True
        if schedule.kind == "seconds":
            reference = watch.last_log_time
            if reference is None:
                reference = watch.baseline_time
            if reference is None:
                return False
            return now - reference >= schedule.interval
        counter = snapshot.get(schedule.key)
        if counter is None:
            return False
        if watch.next_threshold is None:
            self._prime_schedule(watch, snapshot)
            return False
        if watch.last_counter_value is not None and counter < watch.last_counter_value:
            # the counter was reset (empty(), state restore, ...): re-baseline
            # both the counters and the clock so the next rate only covers the
            # post-reset window.
            watch.baseline_counters = dict(snapshot)
            watch.baseline_time = now
            watch.next_threshold = (
                counter // schedule.interval + 1
            ) * schedule.interval
            watch.last_counter_value = counter
            return False
        watch.last_counter_value = counter
        if counter >= watch.next_threshold:
            watch.next_threshold = (
                counter // schedule.interval + 1
            ) * schedule.interval
            return True
        return False

    def _prime_schedule(self, watch: _Watch, snapshot: dict[str, float]) -> None:
        schedule = watch.schedule
        if schedule is not None and schedule.kind == "counter":
            counter = snapshot.get(schedule.key)
            if counter is not None:
                watch.next_threshold = (
                    counter // schedule.interval + 1
                ) * schedule.interval
                watch.last_counter_value = counter

    def _compute_rates(
        self, watch: _Watch, snapshot: dict[str, float], now: float
    ) -> dict[str, float]:
        if watch.baseline_time is None:
            return {}
        elapsed = now - watch.baseline_time
        if elapsed <= 0:
            return {}
        if watch.rate_keys is not None:
            keys = watch.rate_keys
        else:
            keys = tuple(key for key in snapshot if key in _DEFAULT_RATE_NAMES)
        rates = {}
        for key in keys:
            current = snapshot.get(key)
            previous = watch.baseline_counters.get(key)
            if current is None or previous is None:
                continue
            delta = current - previous
            if delta < 0:
                continue
            rate_name = _DEFAULT_RATE_NAMES.get(key, f"{key}_per_second")
            rates[rate_name] = delta / elapsed
        return rates
