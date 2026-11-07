# Logger

torchrl.record.loggers.Logger(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/common.html#Logger)

A template for loggers.

Keyword Arguments:

- **service_backend** (*str*) - Deployment backend. One of `"direct"`,
`"process"`, or `"ray"`. Defaults to `"direct"`.
- **service_backend_options** (*dict**,**optional*) - Backend options. Process
services accept `context`/`mp_context`, `max_queue_size`, and
`startup_timeout`. Ray services accept `actor_options` and
`ray_init_config`.
- **use_ray_service** (*bool*) - If `True`, the logger runs as a Ray actor
in a separate process. Deprecated in favor of
`service_backend="ray"` and scheduled for removal in v0.16.
Defaults to `False`.
- **ray_actor_options** (*dict**,**optional*) - Options passed to `ray.remote()`
when creating the Ray actor (e.g., `{"num_cpus": 1}`).
Only used when `use_ray_service=True`.