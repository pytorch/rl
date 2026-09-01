# Service

*class*torchrl.services.Service(**args*, ***kwargs*)[[source]](../../_modules/torchrl/services/base.html#Service)

Owner-side contract for a long-lived TorchRL service.

A service owns lifecycle and heavy resources. `client()` returns the
lightweight capability that may be passed to worker processes or actors;
that client intentionally has no `start` or `shutdown` methods.

Examples

```
>>> from torchrl.record.loggers import CSVLogger
>>> logger = CSVLogger(exp_name="example", log_dir="/tmp")
>>> _ = logger.start()
>>> logger.client() is logger
True
>>> logger.shutdown()
```

client() → ClientT[[source]](../../_modules/torchrl/services/base.html#Service.client)

Return a cheap, picklable, capability-restricted client.

*property*is_alive*: bool*

Whether the owned service is running.

shutdown(*timeout: float | None = None*) → None[[source]](../../_modules/torchrl/services/base.html#Service.shutdown)

Stop the owned service and release its resources.

start() → Self[[source]](../../_modules/torchrl/services/base.html#Service.start)

Start the owned service and return `self`.