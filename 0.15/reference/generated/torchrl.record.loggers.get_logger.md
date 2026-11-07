# get_logger

torchrl.record.loggers.get_logger(*logger_type: Literal['tensorboard', 'csv', 'wandb', 'mlflow', 'trackio', ''] | None*, *logger_name: str*, *experiment_name: str*, ***, *state_dict: Mapping[str, Any] | None = None*, *service_backend: Literal['direct', 'process', 'ray'] = 'direct'*, *service_backend_options: dict[str, Any] | None = None*, *use_ray_service: bool = False*, *ray_actor_options: dict[str, Any] | None = None*, ***kwargs*) → Logger | None[[source]](../../_modules/torchrl/record/loggers/utils.html#get_logger)

Get a logger instance of the provided logger_type.

Parameters:

- **logger_type** (*str*) - One of tensorboard / csv / wandb / mlflow / trackio.
If empty, `None` is returned.
- **logger_name** (*str*) - Name to be used as a log_dir
- **experiment_name** (*str*) - Name of the experiment

Keyword Arguments:

- **state_dict** (*Mapping**[**str**,**Any**] or**None**,**optional*) - Saved logger state from
`state_dict()`. Restores the saved
name, directory and counters for CSV/TensorBoard, or resumes the
saved W&B run with strict `resume="must"` semantics. Other logger
types currently reject this option before opening a service.
Defaults to `None` (create a logger normally).
- **service_backend** - One of `"direct"`, `"process"`, or `"ray"`.
- **service_backend_options** - Process or Ray initialization options.
- **use_ray_service** - Deprecated compatibility flag for the Ray backend.
- **ray_actor_options** - Deprecated spelling for Ray actor options.
- ****kwargs** - May contain `wandb_kwargs`, `mlflow_kwargs`, or
`trackio_kwargs`.