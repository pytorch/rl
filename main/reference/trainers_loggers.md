# Loggers

Logger classes for experiment tracking and visualization.

| [`Logger`](generated/torchrl.record.loggers.Logger.html#torchrl.record.loggers.Logger)(*args[, use_ray_service]) | A template for loggers. |
| --- | --- |
| [`csv.CSVLogger`](generated/torchrl.record.loggers.csv.CSVLogger.html#torchrl.record.loggers.csv.CSVLogger)(*args[, use_ray_service]) | A minimal-dependency CSV logger. |
| [`mlflow.MLFlowLogger`](generated/torchrl.record.loggers.mlflow.MLFlowLogger.html#torchrl.record.loggers.mlflow.MLFlowLogger)(*args[, use_ray_service]) | Wrapper for the mlflow logger. |
| [`tensorboard.TensorboardLogger`](generated/torchrl.record.loggers.tensorboard.TensorboardLogger.html#torchrl.record.loggers.tensorboard.TensorboardLogger)(*args[, ...]) | Wrapper for the Tensoarboard logger. |
| [`trackio.TrackioLogger`](generated/torchrl.record.loggers.trackio.TrackioLogger.html#torchrl.record.loggers.trackio.TrackioLogger)(*args[, use_ray_service]) | Wrapper for the trackio logger. |
| [`wandb.WandbLogger`](generated/torchrl.record.loggers.wandb.WandbLogger.html#torchrl.record.loggers.wandb.WandbLogger)(*args[, use_ray_service]) | Wrapper for the wandb logger. |
| [`get_logger`](generated/torchrl.record.loggers.get_logger.html#torchrl.record.loggers.get_logger)(logger_type, logger_name, ...) | Get a logger instance of the provided logger_type. |
| [`generate_exp_name`](generated/torchrl.record.loggers.generate_exp_name.html#torchrl.record.loggers.generate_exp_name)(model_name, experiment_name) | Generates an ID (str) for the described experiment using UUID and current date. |

## Recording utils

| [`VideoRecorder`](generated/torchrl.record.VideoRecorder.html#torchrl.record.VideoRecorder)(logger, tag[, in_keys, skip, ...]) | Video Recorder transform. |
| --- | --- |
| [`TensorDictRecorder`](generated/torchrl.record.TensorDictRecorder.html#torchrl.record.TensorDictRecorder)(out_file_base[, ...]) | TensorDict recorder. |
| [`PixelRenderTransform`](generated/torchrl.record.PixelRenderTransform.html#torchrl.record.PixelRenderTransform)([out_keys, preproc, ...]) | A transform to call render on the parent environment and register the pixel observation in the tensordict. |