# MujocoPlaygroundWrapper

torchrl.envs.MujocoPlaygroundWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/mujoco_playground.html#MujocoPlaygroundWrapper)

Google DeepMind MuJoCo Playground environment wrapper.

MuJoCo Playground is a collection of JAX-based MJX environments spanning
locomotion, manipulation, and dm_control suite tasks.

GitHub: [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)

Parameters:

- **env** (*mujoco_playground._src.mjx_env.MjxEnv*) - the environment to wrap.
- **agent_mapping** ([`MujocoPlaygroundAgentMapping`](torchrl.envs.MujocoPlaygroundAgentMapping.html#torchrl.envs.MujocoPlaygroundAgentMapping) or str, optional) - if provided, the environment is decomposed into a cooperative
multi-agent task. Can be either a [`MujocoPlaygroundAgentMapping`](torchrl.envs.MujocoPlaygroundAgentMapping.html#torchrl.envs.MujocoPlaygroundAgentMapping)
instance or a string key into `KNOWN_MARL_MAPPINGS`.
Known string values: `"ant_4x2"`, `"halfcheetah_6x1"`,
`"hopper_3x1"`, `"humanoid_9|8"`, `"walker2d_2x3"`.
Defaults to `None` (single-agent mode).

Keyword Arguments:

- **from_pixels** (*bool**,**optional*) - Not yet supported.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `torch.device("cpu")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
In `mujoco_playground`, this controls the number of environments
simulated in parallel via JAX's `vmap` on a single device (GPU/TPU).
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.

Variables:

**available_envs** - environments available to build

Note

Unlike [`BraxWrapper`](torchrl.envs.BraxWrapper.html#torchrl.envs.BraxWrapper), this wrapper does **not**
copy the underlying JAX env state into the output `TensorDict`. The
state is kept on the env instance (`self._current_state`) and rolled
forward by `_step`; this avoids round-tripping MJX/pytree state
through `TensorDict`, which would break MJX's metadata pytree
registration. As a consequence, the output `TensorDict` only
contains `observation` (or per-key obs for dict-obs envs),
`reward`, `done` and `terminated` -- there is no `state` key.

Warning

Because the JAX state is held on the instance rather than carried in
the `TensorDict`, **partial resets are not supported**: any call to
[`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) re-initialises the *entire* vmapped batch, ignoring the
`"_reset"` mask. For a `batch_size` greater than one whose
sub-environments terminate at different steps (e.g. early-terminating
locomotion tasks driven by a data collector), prefer scaling with
`num_workers` (one scalar env per worker) over a single large
vmapped `batch_size`. This matches the behaviour of
[`BraxWrapper`](torchrl.envs.BraxWrapper.html#torchrl.envs.BraxWrapper).

Note

`terminated` is set equal to `done`; this wrapper does not expose a
separate time-limit `truncated` signal. For finite-horizon tasks
where bootstrapping at the episode boundary matters, append a
[`StepCounter`](torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) (with `max_steps`) or
otherwise track truncations yourself.

Examples

```
>>> from mujoco_playground import dm_control_suite
>>> from torchrl.envs import MujocoPlaygroundWrapper
>>> import torch
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> base_env = dm_control_suite.load("CartpoleBalance")
>>> env = MujocoPlaygroundWrapper(base_env, device=device)
>>> env.set_seed(0)
>>> td = env.reset()
>>> td["action"] = env.action_spec.rand()
>>> td = env.step(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(torch.Size([1]), dtype=torch.float32),
 done: Tensor(torch.Size([1]), dtype=torch.bool),
 next: TensorDict(
 fields={
 done: Tensor(torch.Size([1]), dtype=torch.bool),
 observation: Tensor(torch.Size([5]), dtype=torch.float32),
 reward: Tensor(torch.Size([1]), dtype=torch.float32),
 terminated: Tensor(torch.Size([1]), dtype=torch.bool)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 observation: Tensor(torch.Size([5]), dtype=torch.float32),
 terminated: Tensor(torch.Size([1]), dtype=torch.bool)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> print(env.available_envs)
['AcrobotSwingup', 'AcrobotSwingupSparse', 'BallInCupCatch', ...]
```