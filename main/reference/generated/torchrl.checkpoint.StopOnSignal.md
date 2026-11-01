# StopOnSignal

*class*torchrl.checkpoint.StopOnSignal(*signals: Collection[int] = (Signals.SIGINT, Signals.SIGTERM)*, ***, *on_request: Callable[[str], None] | None = None*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#StopOnSignal)

Turn termination signals into a stop request checked at loop boundaries.

The first handled signal records a request so a training loop can finish
the current batch, save a checkpoint and exit cleanly. A second signal
raises `KeyboardInterrupt` so a loop that cannot reach a boundary
can still be interrupted. Handlers are installed when the context is
entered and the previous handlers are restored on exit. Python only
accepts signal handlers in the main thread, so entering the context from
another thread leaves the handlers untouched and logs a warning.

Parameters:

- **signals** (*Collection**[**int**]**,**optional*) - signal numbers to handle.
Defaults to `SIGINT` and `SIGTERM`.
- **on_request** (*Callable**[**[**str**]**,**None**]**,**optional*) - callback invoked with the
signal name when the first signal arrives.

Examples

```
>>> from torchrl.checkpoint import StopOnSignal
>>> with StopOnSignal() as stop:
... for _ in range(3):
... if stop.requested:
... break
>>> stop.requested
False
```

*property*requested*: bool*

Whether a handled signal has been received.