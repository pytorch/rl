# PrefixLogger

torchrl.record.loggers.PrefixLogger(*logger: LoggerT*, *prefix: str*)[[source]](../../_modules/torchrl/record/loggers/common.html#PrefixLogger)

A namespaced view over an existing logger.

Metric, video, histogram, and string names are prefixed consistently while
hyperparameter keys are forwarded unchanged. Chained views compose their
prefixes, and lifecycle, state, and experiment access remain owned by the
wrapped logger. A view over an owning [`Logger`](torchrl.record.loggers.Logger.html#torchrl.record.loggers.Logger) is accepted wherever
a `Logger` instance is required; a view over a service client retains the
client's restricted capabilities.

Parameters:

- **logger** - Logger or logger service client to wrap.
- **prefix** - Non-empty namespace to prepend to logged names. Leading and
trailing `/` characters are ignored.

Examples

```
>>> from torchrl.record.loggers import CSVLogger
>>> logger = CSVLogger(exp_name="run", log_dir="/tmp")
>>> training = logger.with_prefix("training")
>>> training.log_scalar("loss", 1.0, step=0)
>>> logger.close()
```