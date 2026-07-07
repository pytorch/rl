# TensorboardLogger

torchrl.record.loggers.tensorboard.TensorboardLogger(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/tensorboard.html#TensorboardLogger)

Wrapper for the Tensoarboard logger.

See also `TensorboardLoggerConfig`.

Parameters:

- **exp_name** (*str*) - The name of the experiment.
- **log_dir** (*str*) - the tensorboard log_dir. Defaults to `td_logs`.