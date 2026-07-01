# Logger

torchrl.record.loggers.Logger(**args*, *use_ray_service=False*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/common.html#Logger)

A template for loggers.

Keyword Arguments:

- **use_ray_service** (*bool*) - If `True`, the logger runs as a Ray actor
in a separate process. All method calls are delegated to the remote
actor via `ray.get(actor.method.remote(...))`. CUDA tensors in
`log_metrics()` and `log_video()` are automatically moved
to CPU before the remote call. Requires `ray` to be installed.
Default: `False`.
- **ray_actor_options** (*dict**,**optional*) - Options passed to `ray.remote()`
when creating the Ray actor (e.g., `{"num_cpus": 1}`).
Only used when `use_ray_service=True`.