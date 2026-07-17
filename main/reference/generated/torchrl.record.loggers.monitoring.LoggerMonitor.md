# LoggerMonitor

torchrl.record.loggers.monitoring.LoggerMonitor(*logger: Logger*, ***, *poll_interval: float = 1.0*, *background: bool = True*, *on_error: Literal['warn', 'raise', 'ignore'] = 'warn'*) → None[[source]](../../_modules/torchrl/record/loggers/monitoring.html#LoggerMonitor)

A pull-based monitor logging operational statistics of collectors and replay buffers.

The monitor periodically requests a cheap `stats()` snapshot from each
watched object, decides whether the snapshot is due for logging according
to the per-target schedule, derives rates from cumulative counter deltas,
namespaces the metrics as `"<name>/<metric>"` and forwards them to a
single `Logger`.

Because snapshots are pulled from the monitor's own thread (or from
explicit `step()` calls), no logging work is ever executed on the
collection, write or sampling hot paths of the watched objects, and a
slow logging backend can only delay the next poll, never build an
unbounded backlog. The monitor works with any object exposing a
`stats()` method returning a flat mapping of scalars: in particular
local, multiprocessing and Ray collectors and replay buffers share the
same interface.

The monitor never takes ownership of the logger or of the watched
objects: stopping the monitor leaves both running, and shutting down the
logger remains the caller's responsibility.

Parameters:

**logger** (*Logger*) - the output sink. Any TorchRL logger works, including
loggers running as a service (`service_backend="ray"`).

Keyword Arguments:

- **poll_interval** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - how often the background thread
polls the watched objects, in seconds. Counter schedules are
evaluated at each poll, so this bounds their resolution.
Defaults to `1.0`.
- **background** (*bool**,**optional*) - if `True` (default), entering the
monitor context (or calling `start()`) spawns a daemon
thread that polls every `poll_interval` seconds. If `False`,
nothing runs in the background and the user drives the monitor
through explicit `step()` calls, which keeps tests and
deterministic loops reproducible.
- **on_error** (*str**,**optional*) - behavior when polling or logging a target
fails: `"warn"` (default) logs a warning once per failure
streak through the torchrl logger, `"ignore"` silently skips,
and `"raise"` propagates the exception (in background mode
this terminates the monitor thread).

Examples

```
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
```

The typical background usage mirrors the RFC acceptance example:

```
>>> with LoggerMonitor(logger) as monitor: 
... monitor.watch(collector, name="collector", schedule=Every.counter("frames", 10_000))
... monitor.watch(replay_buffer, name="replay_buffer", schedule=Every.seconds(5), step="write_count")
... collector.start()
... run_training()
```