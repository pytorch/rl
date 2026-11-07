# Every

torchrl.record.loggers.monitoring.Every(*kind: Literal['seconds', 'counter']*, *interval: float*, *key: str | None = None*) → None[[source]](../../_modules/torchrl/record/loggers/monitoring.html#Every)

A logging schedule for [`LoggerMonitor`](torchrl.record.loggers.monitoring.LoggerMonitor.html#torchrl.record.loggers.monitoring.LoggerMonitor).

Schedules are built through the two factory methods rather than the
constructor:

- `Every.seconds()` triggers on wall-clock time;
- `Every.counter()` triggers whenever a cumulative counter reported
by the watched object's `stats()` snapshot crosses a multiple of
`interval`.

Counter schedules are observed, not executed at the exact operation: if
a counter jumps across several thresholds between two polls, the latest
snapshot is logged once. A counter decrease (after a reset or a state
restoration) re-baselines the schedule instead of producing spurious
logs.

Examples

```
>>> import tempfile
>>> import torch
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.record import CSVLogger
>>> from torchrl.record.loggers.monitoring import Every, LoggerMonitor
>>> logger = CSVLogger(exp_name="every_demo", log_dir=tempfile.mkdtemp())
>>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
>>> monitor = LoggerMonitor(logger, background=False)
>>> name = monitor.watch(
... rb, name="rb", schedule=Every.counter("write_count", 50), log_on_start=False
... )
>>> logged = monitor.step() # primes the schedule, nothing logged yet
>>> print(logged)
{}
>>> _ = rb.extend(torch.arange(60))
>>> logged = monitor.step() # write_count crossed the 50 threshold
>>> print(logged["rb"]["rb/write_count"])
60.0
```