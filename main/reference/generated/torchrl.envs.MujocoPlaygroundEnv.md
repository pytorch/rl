# MujocoPlaygroundEnv

torchrl.envs.MujocoPlaygroundEnv(**args*, *num_workers: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/mujoco_playground.html#MujocoPlaygroundEnv)

Google DeepMind MuJoCo Playground environment wrapper built with the environment name.

MuJoCo Playground is a collection of JAX-based MJX environments spanning
locomotion, manipulation, and dm_control suite tasks. All environments from
all suites are accessible by name via the unified registry.

GitHub: [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)

Parameters:

- **env_name** (*str*) - the environment name of the env to wrap. Must be part of
`available_envs`.
- **config** (*ml_collections.ConfigDict**,**optional*) - configuration for the environment.
If `None`, the default configuration is used. Defaults to `None`.
- **config_overrides** (*dict**,**optional*) - overrides to apply on top of `config`.
Defaults to `None`.
- **agent_mapping** ([`MujocoPlaygroundAgentMapping`](torchrl.envs.MujocoPlaygroundAgentMapping.html#torchrl.envs.MujocoPlaygroundAgentMapping) or str, optional) - if provided, the environment is decomposed into a cooperative
multi-agent task. Can be either a [`MujocoPlaygroundAgentMapping`](torchrl.envs.MujocoPlaygroundAgentMapping.html#torchrl.envs.MujocoPlaygroundAgentMapping)
instance or a string key into `KNOWN_MARL_MAPPINGS`.
Known string values: `"ant_4x2"`, `"halfcheetah_6x1"`,
`"hopper_3x1"`, `"humanoid_9|8"`, `"walker2d_2x3"`.
The mapping and the environment name are validated against each other
at construction time. Defaults to `None` (single-agent mode).

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
- **num_workers** (*int**,**optional*) - if greater than 1, a lazy [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv)
will be returned instead, with each worker instantiating its own
`MujocoPlaygroundEnv` instance. Defaults to `None`.

Note

There are two orthogonal ways to scale environment throughput:

- **batch_size**: Uses MuJoCo Playground's native JAX-based vectorization
(`vmap`) to run multiple environments in parallel on a single GPU/TPU.
- **num_workers**: Uses TorchRL's [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) to
spawn multiple Python processes, each running its own
`MujocoPlaygroundEnv`.

These can be combined: `MujocoPlaygroundEnv("CartpoleBalance", batch_size=[128], num_workers=4)`
creates 4 worker processes each running 128 vectorized environments.

Variables:

**available_envs** - environments available to build (all suites combined)

Examples

```
>>> from torchrl.envs import MujocoPlaygroundEnv
>>> import torch
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> env = MujocoPlaygroundEnv("CartpoleBalance", device=device)
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

To take advantage of MuJoCo Playground's JAX-based parallelism, pass a
`batch_size` to run multiple environments in parallel on a single device:

Examples

```
>>> from torchrl.envs import MujocoPlaygroundEnv
>>> import torch
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> env = MujocoPlaygroundEnv("CartpoleBalance", batch_size=[128], device=device)
>>> env.set_seed(0)
>>> td = env.rollout(100)
>>> print(td.shape)
torch.Size([128, 100])
```