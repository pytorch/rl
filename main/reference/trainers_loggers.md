# Loggers

Logger classes for experiment tracking and visualization.

Loggers support an owner/client deployment model. The direct backend returns
the logger itself from `client()`. Process and Ray backends return a
picklable client that only exposes logging operations; lifecycle calls remain
on the owner. Remote calls preserve direct-logger semantics: they return after
the concrete logger method has run, propagate service-side errors immediately,
and preserve custom `log_*` return values. Bounded transport queues provide
backpressure when several clients log concurrently.

```
from torchrl.record import CSVLogger

logger = CSVLogger(
 exp_name="run",
 log_dir="logs",
 service_backend="process",
 service_backend_options={"max_queue_size": 256},
)
worker_logger = logger.client()
worker_logger.log_scalar("loss", 1.0, step=0)
logger.flush() # Flush buffers owned by the concrete logging SDK.
logger.shutdown()
```

| [`Logger`](generated/torchrl.record.loggers.Logger.html#torchrl.record.loggers.Logger)(*args[, use_ray_service, ...]) | A template for loggers. |
| --- | --- |
| [`ProcessLogger`](generated/torchrl.record.loggers.ProcessLogger.html#torchrl.record.loggers.ProcessLogger)(logger_cls, *args[, ...]) | Driver-owned logger service running in a dedicated process. |
| [`RayLogger`](generated/torchrl.record.loggers.RayLogger.html#torchrl.record.loggers.RayLogger)(logger_cls, *args[, ...]) | Driver-owned Ray logger service with restricted worker clients. |
| [`csv.CSVLogger`](generated/torchrl.record.loggers.csv.CSVLogger.html#torchrl.record.loggers.csv.CSVLogger)(*args[, use_ray_service, ...]) | A minimal-dependency CSV logger. |
| [`mlflow.MLFlowLogger`](generated/torchrl.record.loggers.mlflow.MLFlowLogger.html#torchrl.record.loggers.mlflow.MLFlowLogger)(*args[, ...]) | Wrapper for the mlflow logger. |
| [`tensorboard.TensorboardLogger`](generated/torchrl.record.loggers.tensorboard.TensorboardLogger.html#torchrl.record.loggers.tensorboard.TensorboardLogger)(*args[, ...]) | Wrapper for the Tensoarboard logger. |
| [`trackio.TrackioLogger`](generated/torchrl.record.loggers.trackio.TrackioLogger.html#torchrl.record.loggers.trackio.TrackioLogger)(*args[, ...]) | Wrapper for the trackio logger. |
| [`wandb.WandbLogger`](generated/torchrl.record.loggers.wandb.WandbLogger.html#torchrl.record.loggers.wandb.WandbLogger)(*args[, use_ray_service, ...]) | Wrapper for the wandb logger. |
| [`get_logger`](generated/torchrl.record.loggers.get_logger.html#torchrl.record.loggers.get_logger)(logger_type, logger_name, ...[, ...]) | Get a logger instance of the provided logger_type. |
| [`generate_exp_name`](generated/torchrl.record.loggers.generate_exp_name.html#torchrl.record.loggers.generate_exp_name)(model_name, experiment_name) | Generates an ID (str) for the described experiment using UUID and current date. |

## Recording utils

| [`VideoRecorder`](generated/torchrl.record.VideoRecorder.html#torchrl.record.VideoRecorder)(logger, tag[, in_keys, skip, ...]) | Video Recorder transform. |
| --- | --- |
| [`TensorDictRecorder`](generated/torchrl.record.TensorDictRecorder.html#torchrl.record.TensorDictRecorder)(out_file_base[, ...]) | TensorDict recorder. |
| [`PixelRenderTransform`](generated/torchrl.record.PixelRenderTransform.html#torchrl.record.PixelRenderTransform)([out_keys, preproc, ...]) | A transform to call render on the parent environment and register the pixel observation in the tensordict. |