# sync_async_collector

torchrl.trainers.helpers.sync_async_collector(*env_fns: Callable | list[Callable]*, *env_kwargs: dict | list[dict] | None*, *num_env_per_collector: int | None = None*, *num_collectors: int | None = None*, ***kwargs*) → [MultiAsyncCollector](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)[[source]](../../_modules/torchrl/trainers/helpers/collectors.html#sync_async_collector)

Runs asynchronous collectors, each running synchronous environments.

[![../../_images/aafig-bbb875b648162ef711030ef56c3eec3894efb3ed.svg](../../_images/aafig-bbb875b648162ef711030ef56c3eec3894efb3ed.svg)](../../_images/aafig-bbb875b648162ef711030ef56c3eec3894efb3ed.svg)

Environment types can be identical or different. In the latter case, env_fns should be a list with all the creator
fns for the various envs,
and the policy should handle those envs in batch.

Parameters:

- **env_fns** - Callable (or list of Callables) returning an instance of EnvBase class.
- **env_kwargs** - Optional. Dictionary (or list of dictionaries) containing the kwargs for the environment being created.
- **num_env_per_collector** - Number of environments per data collector. The product
num_env_per_collector * num_collectors should be less or equal to the number of workers available.
- **num_collectors** - Number of data collectors to be run in parallel.
- ****kwargs** - Other kwargs passed to the data collectors