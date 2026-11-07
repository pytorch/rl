# GymWrapper

torchrl.envs.GymWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/gym.html#GymWrapper)

OpenAI Gym environment wrapper.

Works across [gymnasium](https://gymnasium.farama.org/) and [OpenAI/gym](https://github.com/openai/gym).

Parameters:

- **env** (*gym.Env*) - the environment to wrap. Batched environments (`VecEnv`
or `gym.VectorEnv`) are supported and the environment batch-size
will reflect the number of environments executed in parallel.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.

Keyword Arguments:

- **from_pixels** (*bool**,**optional*) - if `True`, an attempt to return the pixel
observations from the env will be performed. By default, these observations
will be written under the `"pixels"` entry.
The method being used varies
depending on the gym version and may involve a `wrappers.pixel_observation.PixelObservationWrapper`.
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
- **convert_actions_to_numpy** (*bool**,**optional*) - if `True`, actions will be
converted from tensors to numpy arrays and moved to CPU before being passed to the
env step function. Set this to `False` if the environment is evaluated
on GPU, such as IsaacLab.
Defaults to `True`.
- **missing_obs_value** (*Any**,**optional*) - default value to use as placeholder for missing observations, when
the environment is auto-resetting and missing observations cannot be found in the info dictionary
(e.g., with IsaacLab). This argument is passed to `VecGymEnvTransform` by
the metaclass.
- **native_autoreset** (*bool**,**optional*) - if `True` and the wrapped environment
is an Isaac Lab vectorized environment, uses the native auto-reset
observation returned by the environment as the next root observation
instead of calling reset from
`step_and_maybe_reset()`. The terminal
`"next"` observation is still invalid and filled with `NaN` for
floating point observations. Defaults to `False`.

Variables:

**available_envs** (*List**[**str**]*) - a list of environments to build.

Note

If an attribute cannot be found, this class will attempt to retrieve it from
the nested env:

```
>>> from torchrl.envs import GymWrapper
>>> import gymnasium as gym
>>> env = GymWrapper(gym.make("Pendulum-v1"))
>>> print(env.spec.max_episode_steps)
200
```

Examples

```
>>> import gymnasium as gym
>>> from torchrl.envs import GymWrapper
>>> base_env = gym.make("Pendulum-v1")
>>> env = GymWrapper(base_env)
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

Note

info dictionaries will be read using `default_info_dict_reader`
if no other reader is provided. To provide another reader, refer to
`set_info_dict_reader()`. To automatically register the info_dict
content, refer to [`torchrl.envs.GymLikeEnv.auto_register_info_dict()`](torchrl.envs.GymLikeEnv.html#torchrl.envs.GymLikeEnv.auto_register_info_dict).
For parallel (Vectorized) environments, the info dictionary reader is automatically set and should
not be set manually.

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