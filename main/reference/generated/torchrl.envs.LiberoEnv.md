# LiberoEnv

torchrl.envs.LiberoEnv(**args*, *num_workers: int | None = None*, *num_envs: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/libero.html#LiberoEnv)

LIBERO environment built from a task-suite name and task id.

GitHub: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

Paper: [https://arxiv.org/abs/2306.03310](https://arxiv.org/abs/2306.03310) (LIBERO: Benchmarking Knowledge
Transfer for Lifelong Robot Learning, Liu et al., 2023)

See [`LiberoWrapper`](torchrl.envs.LiberoWrapper.html#torchrl.envs.LiberoWrapper) for the full TensorDict schema
and keyword arguments. This constructor builds the underlying
`OffScreenRenderEnv` from the benchmark registry, fetches the task's
language instruction and its fixed initial states (50 per task in the
standard suites) and wires them into the init-state control machinery.

Parameters:

- **task_suite** (*str*) - the task-suite name, from `available_envs`
(e.g. `"libero_spatial"`, `"libero_object"`,
`"libero_goal"`, `"libero_10"`).
- **task_id** (*int*) - the task index inside the suite.

Keyword Arguments:

- **num_workers** (*int**,**optional*) - if greater than `1`, return a
[`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) with `num_workers` LIBERO
workers. Defaults to `1`.
- **num_envs** (*int**,**optional*) - alias for `num_workers`.
- **camera_height** (*int**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - rendered image height.
When `num_workers > 1`, a list dispatches one value per worker
and must have length `num_workers`. Defaults to `256` (the
SimpleVLA-RL / OpenVLA-OFT resolution).
- **camera_width** (*int**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - rendered image width.
When `num_workers > 1`, a list dispatches one value per worker
and must have length `num_workers`. Defaults to `256`.
- **render_gpu_device_id** (*int**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - GPU device id used by
robosuite for offscreen rendering. This is the EGL-visible device
id inside the process/container, not necessarily the global CUDA
ordinal. Use this to spread LIBERO render workers across multiple
GPUs, e.g. `worker_idx % num_render_gpus`. If omitted, robosuite
selects its default device. When `num_workers > 1`, a list
dispatches one value per worker and must have length
`num_workers`. Defaults to `None`.
- **env_kwargs** (*dict**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**dict**,**optional*) - extra keyword arguments
forwarded to `OffScreenRenderEnv` (e.g. `horizon`). When
`num_workers > 1`, a list dispatches one dict per worker and
must have length `num_workers`.
- ****kwargs** - see [`LiberoWrapper`](torchrl.envs.LiberoWrapper.html#torchrl.envs.LiberoWrapper). When
`num_workers > 1`, list-valued keyword arguments are dispatched
across workers and must have length `num_workers`; use tuples for
sequence-valued arguments that should be broadcast to every worker
(for example `proprio_keys`).

Examples

```
>>> from torchrl.envs import LiberoEnv
>>> env = LiberoEnv(
... "libero_spatial",
... task_id=0,
... camera_height=128,
... camera_width=128,
... init_state_mode="cycle",
... )
>>> td = env.reset()
>>> td["language_instruction"]
'pick up the black bowl between the plate and the ramekin and place it on the plate'
>>> td["observation", "image"].shape
torch.Size([3, 128, 128])
```