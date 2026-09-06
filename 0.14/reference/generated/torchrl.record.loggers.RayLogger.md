# RayLogger

torchrl.record.loggers.RayLogger(*logger_cls: type[LoggerT]*, **args: Any*, *ray_actor_options: dict[str, Any] | None = None*, *ray_init_config: dict[str, Any] | None = None*, ***kwargs: Any*) → None[[source]](../../_modules/torchrl/record/loggers/ray.html#RayLogger)

Driver-owned Ray logger service with restricted worker clients.

Existing direct construction and `use_ray_service=True` continue to
create this owner. Use `client()` before sending the logger to workers.

Parameters:

- **logger_cls** - Concrete [`Logger`](torchrl.record.loggers.Logger.html#torchrl.record.loggers.Logger) class.
- ***args** - Positional arguments forwarded to `logger_cls`.
- **ray_actor_options** - Options used to construct the Ray actor.
- **ray_init_config** - Options used to initialize Ray when needed.
- ****kwargs** - Keyword arguments forwarded to `logger_cls`.

Examples

```
>>> from torchrl.record import CSVLogger, RayLogger
>>> logger = RayLogger(CSVLogger, exp_name="run", log_dir="/tmp") 
>>> client = logger.client() 
>>> client.log_scalar("loss", 1.0, step=0) 
>>> logger.shutdown()
```