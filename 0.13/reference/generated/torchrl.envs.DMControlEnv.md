# DMControlEnv

torchrl.envs.DMControlEnv(**args*, *num_workers: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/dm_control.html#DMControlEnv)

DeepMind Control lab environment wrapper.

The DeepMind control library can be found here: [deepmind/dm_control](https://github.com/deepmind/dm_control).

Paper: [https://arxiv.org/abs/2006.12983](https://arxiv.org/abs/2006.12983)

Parameters:

- **env_name** (*str*) - name of the environment.
- **task_name** (*str*) - name of the task.
- **num_workers** (*int**,**optional*) - number of parallel environments. Defaults to 1.
When `num_workers > 1`, a lazy [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) is
returned instead of a single environment. The parallel environment
is not started until it is actually used (e.g., via reset/step or
accessing specs). Use `configure_parallel()`
to set parallel execution parameters before the environment starts.

Keyword Arguments:

- **from_pixels** (*bool**,**optional*) - if `True`, an attempt to return the pixel
observations from the env will be performed.
By default, these observations
will be written under the `"pixels"` entry.
Defaults to `False`.
- **pixels_only** (*bool**,**optional*) - if `True`, only the pixel observations will
be returned (by default under the `"pixels"` entry in the output tensordict).
If `False`, observations (eg, states) and pixels will be returned
whenever `from_pixels=True`. Defaults to `True`.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `torch.device("cpu")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
Should match the leading dimensions of all observations, done states,
rewards, actions and infos.
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.

Variables:

**available_envs** (*list*) - a list of `Tuple[str, List[str]]` representing the
environment / task pairs available.

Examples

```
>>> from torchrl.envs import DMControlEnv
>>> env = DMControlEnv(env_name="cheetah", task_name="run",
... from_pixels=True, frame_skip=4)
>>> td = env.rand_step()
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([6]), device=cpu, dtype=torch.float64, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 pixels: Tensor(shape=torch.Size([240, 320, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 position: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float64, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float64, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 velocity: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.float64, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> print(env.available_envs)
[('acrobot', ['swingup', 'swingup_sparse']), ...]
>>> # For running multiple envs in parallel (returns a lazy ParallelEnv)
>>> env = DMControlEnv("cheetah", "run", num_workers=4)
>>> # Configure parallel parameters before the env starts
>>> env.configure_parallel(use_buffers=True, num_threads=2)
>>> # Environment starts when first used
>>> env.reset()
```