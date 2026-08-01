# get_logger

torchrl.record.loggers.get_logger(*logger_type: str*, *logger_name: str*, *experiment_name: str*, ***kwargs*) → Logger[[source]](../../_modules/torchrl/record/loggers/utils.html#get_logger)

Get a logger instance of the provided logger_type.

Parameters:

- **logger_type** (*str*) - One of tensorboard / csv / wandb / mlflow / trackio.
If empty, `None` is returned.
- **logger_name** (*str*) - Name to be used as a log_dir
- **experiment_name** (*str*) - Name of the experiment
- **kwargs** (*dict**[**str**]*) - might contain either wandb_kwargs, mlflow_kwargs or trackio_kwargs.
Additionally supports `use_ray_service` (bool) and `ray_actor_options` (dict)
to run the logger as a Ray actor.