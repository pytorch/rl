# ProcessLogger

torchrl.record.loggers.ProcessLogger(*logger_cls: type[LoggerT]*, **args: Any*, *mp_context: str | mp.context.BaseContext | None = None*, *max_queue_size: int = 1000*, *startup_timeout: float = 60.0*, ***kwargs: Any*) → None[[source]](../../_modules/torchrl/record/loggers/process.html#ProcessLogger)

Driver-owned logger service running in a dedicated process.

The concrete logger is constructed once in the child process. Worker-side
clients can only submit `log_*` calls; only this owner can flush or stop
the service.

Parameters:

- **logger_cls** - Concrete [`Logger`](torchrl.record.loggers.Logger.html#torchrl.record.loggers.Logger) class.
- ***args** - Positional arguments passed to `logger_cls`.
- **mp_context** - Multiprocessing context or start-method name. Defaults to
`"spawn"`.
- **max_queue_size** - Maximum number of pending logging commands. Defaults
to `1000`.
- **startup_timeout** - Seconds to wait for logger construction. Defaults to
`60`.
- ****kwargs** - Keyword arguments passed to `logger_cls`.

Examples

```
>>> from torchrl.record.loggers import CSVLogger, ProcessLogger
>>> logger = ProcessLogger(CSVLogger, exp_name="run", log_dir="/tmp")
>>> worker_logger = logger.client()
>>> worker_logger.log_scalar("loss", 1.0, step=0)
>>> logger.shutdown()
```