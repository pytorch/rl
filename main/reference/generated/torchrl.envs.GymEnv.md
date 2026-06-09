# GymEnv

torchrl.envs.GymEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/gym.html#GymEnv)

OpenAI Gym environment wrapper constructed by environment ID directly.

Works across [gymnasium](https://gymnasium.farama.org/) and [OpenAI/gym](https://github.com/openai/gym).

Parameters:

- **env_name** (*str*) - the environment id registered in gym.registry.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.

Keyword Arguments:

- **num_envs** (*int**,**optional*) - the number of envs to run in parallel. Defaults to
`None` (a single env is to be run). `AsyncVectorEnv`
will be used by default.
- **num_workers** (*int**,**optional*) - number of top-level worker subprocesses used to create/run
multiple `GymEnv` instances in parallel (handled by the metaclass
`_GymAsyncMeta`). When `num_workers > 1`, a lazy
[`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) is returned whose factory preserves the original
GymEnv kwargs. You can modify the ParallelEnv construction/configuration before
it starts by calling `configure_parallel()`
on the returned object (for example: `env.configure_parallel(use_buffers=True, num_threads=2)`).
When both `num_workers` and `num_envs` are greater than 1, the total number of
environments executed in parallel is `num_workers * num_envs`. Defaults to `1`.
- **disable_env_checker** (*bool**,**optional*) - for gym > 0.24 only. If `True` (default
for these versions), the environment checker won't be run.
- **from_pixels** (*bool**,**optional*) - if `True`, an attempt to return the pixel
observations from the env will be performed. By default, these observations
will be written under the `"pixels"` entry.
The method being used varies
depending on the gym version and may involve a `wrappers.pixel_observation.PixelObservationWrapper`.
Defaults to `False`.
- **pixels_only** (*bool**,**optional*) - if `True`, only the pixel observations will
be returned (by default under the `"pixels"` entry in the output tensordict).
If `False`, observations (eg, states) and pixels will be returned
whenever `from_pixels=True`. Defaults to `False`.
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

**available_envs** (*List**[**str**]*) - the list of envs that can be built.

Note

If an attribute cannot be found, this class will attempt to retrieve it from
the nested env:

```
>>> from torchrl.envs import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> print(env.spec.max_episode_steps)
200
```

If a use-case is not covered by TorchRL, please submit an issue on GitHub.

Examples

```
>>> from torchrl.envs import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> td = env.rand_step()
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> print(env.available_envs)
['ALE/Adventure-ram-v5', 'ALE/Adventure-v5', 'ALE/AirRaid-ram-v5', 'ALE/AirRaid-v5', 'ALE/Alien-ram-v5', 'ALE/Alien-v5',
```

To run multiple environments in parallel:
>>> from torchrl.envs import GymEnv
>>> env = GymEnv("Pendulum-v1", num_workers=4)
>>> td_reset = env.reset()
>>> td = env.rand_step(td_reset)
>>> print(td)
TensorDict(

> fields={
> 
> action: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
> done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
> next: TensorDict(
> 
> 
> 
> 
> > fields={
> > 
> > done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
> > observation: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
> > reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
> > terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
> > truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
> > 
> > 
> > 
> > 
> > 
> > 
> > batch_size=torch.Size([4]),
> > device=None,
> > is_shared=False),
> 
> 
> 
> 
> observation: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
> terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
> truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
> 
> 
> 
> 
> 
> 
> batch_size=torch.Size([4]),
> device=None,
> is_shared=False)

Note

If both OpenAI/gym and gymnasium are present in the virtual environment,
one can swap backend using `set_gym_backend()`:

```
>>> from torchrl.envs import set_gym_backend, GymEnv
>>> with set_gym_backend("gym"):
... env = GymEnv("Pendulum-v1")
... print(env._env)
<class 'gym.wrappers.time_limit.TimeLimit'>
>>> with set_gym_backend("gymnasium"):
... env = GymEnv("Pendulum-v1")
... print(env._env)
<class 'gymnasium.wrappers.time_limit.TimeLimit'>
```

Note

info dictionaries will be read using `default_info_dict_reader`
if no other reader is provided. To provide another reader, refer to
`set_info_dict_reader()`. To automatically register the info_dict
content, refer to [`torchrl.envs.GymLikeEnv.auto_register_info_dict()`](torchrl.envs.GymLikeEnv.html#torchrl.envs.GymLikeEnv.auto_register_info_dict).

Note

Gym spaces are not completely covered.
The following spaces are accounted for provided that they can be represented by a torch.Tensor, a nested tensor
and/or within a tensordict:

- spaces.Box
- spaces.Sequence
- spaces.Tuple
- spaces.Discrete
- spaces.MultiBinary
- spaces.MultiDiscrete
- spaces.Dict

Some considerations should be made when working with gym spaces. For instance, a tuple of spaces
can only be supported if the spaces are semantically identical (same dtype and same number of dimensions).
Ragged dimension can be supported through [`nested_tensor()`](https://docs.pytorch.org/docs/stable/nested.html#torch.nested.nested_tensor), but then there should be only
one level of tuple and data should be stacked along the first dimension (as nested_tensors can only be
stacked along the first dimension).

Check the example in examples/envs/gym_conversion_examples.py to know more!