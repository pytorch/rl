# get_logger

torchrl.record.loggers.get_logger(*logger_type: Literal['tensorboard', 'csv', 'wandb', 'mlflow', 'trackio', ''] | None*, *logger_name: str*, *experiment_name: str*, ***, *service_backend: Literal['direct', 'process', 'ray'] = 'direct'*, *service_backend_options: dict[str, Any] | None = None*, *use_ray_service: bool = False*, *ray_actor_options: dict[str, Any] | None = None*, ***kwargs*) → Logger | None[[source]](../../_modules/torchrl/record/loggers/utils.html#get_logger)

Get a logger instance of the provided logger_type.

Parameters:

- **logger_type** (*str*) - One of tensorboard / csv / wandb / mlflow / trackio.
If empty, `None` is returned.
- **logger_name** (*str*) - Name to be used as a log_dir
- **experiment_name** (*str*) - Name of the experiment
- **service_backend** - One of `"direct"`, `"process"`, or `"ray"`.
- **service_backend_options** - Process or Ray initialization options.
- **use_ray_service** - Deprecated compatibility flag for the Ray backend.
- **ray_actor_options** - Deprecated spelling for Ray actor options.
- ****kwargs** - May contain `wandb_kwargs`, `mlflow_kwargs`, or
`trackio_kwargs`.