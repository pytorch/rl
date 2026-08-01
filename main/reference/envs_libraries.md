# Library Wrappers

TorchRL's mission is to make the training of control and decision algorithm as
easy as it gets, irrespective of the simulator being used (if any).
Multiple wrappers are available for DMControl, Habitat, Jumanji and, naturally,
for Gym.

This last library has a special status in the RL community as being the mostly
used framework for coding simulators. Its successful API has been foundational
and inspired many other frameworks, among which TorchRL.
However, Gym has gone through multiple design changes and it is sometimes hard
to accommodate these as an external adoption library: users usually have their
"preferred" version of the library. Moreover, gym is now being maintained
by another group under the "gymnasium" name, which does not facilitate code
compatibility. In practice, we must consider that users may have a version of
gym *and* gymnasium installed in the same virtual environment, and we must
allow both to work concomittantly.
Fortunately, TorchRL provides a solution for this problem: a special decorator
`set_gym_backend` allows to control which library will be used
in the relevant functions:

```
>>> from torchrl.envs.libs.gym import GymEnv, set_gym_backend, gym_backend
>>> import gymnasium, gym
>>> with set_gym_backend(gymnasium):
... print(gym_backend())
... env1 = GymEnv("Pendulum-v1")
<module 'gymnasium' from '/path/to/venv/python3.10/site-packages/gymnasium/__init__.py'>
>>> with set_gym_backend(gym):
... print(gym_backend())
... env2 = GymEnv("Pendulum-v1")
<module 'gym' from '/path/to/venv/python3.10/site-packages/gym/__init__.py'>
>>> print(env1._env.env.env)
<gymnasium.envs.classic_control.pendulum.PendulumEnv at 0x15147e190>
>>> print(env2._env.env.env)
<gym.envs.classic_control.pendulum.PendulumEnv at 0x1629916a0>
```

We can see that the two libraries modify the value returned by [`gym_backend()`](generated/torchrl.envs.gym_backend.html#torchrl.envs.gym_backend)
which can be further used to indicate which library needs to be used for
the current computation. `set_gym_backend` is also a decorator:
we can use it to tell to a specific function what gym backend needs to be used
during its execution.
The [`gym_backend()`](generated/torchrl.envs.gym_backend.html#torchrl.envs.gym_backend) function allows you to gather
the current gym backend or any of its modules:

```
>>> import mo_gymnasium
>>> with set_gym_backend("gym"):
... wrappers = gym_backend('wrappers')
... print(wrappers)
<module 'gym.wrappers' from '/path/to/venv/python3.10/site-packages/gym/wrappers/__init__.py'>
>>> with set_gym_backend("gymnasium"):
... wrappers = gym_backend('wrappers')
... print(wrappers)
<module 'gymnasium.wrappers' from '/path/to/venv/python3.10/site-packages/gymnasium/wrappers/__init__.py'>
```

Another tool that comes in handy with gym and other external dependencies is
the [`implement_for`](generated/torchrl.implement_for.html#torchrl.implement_for) class. Decorating a function
with `@implement_for` will tell torchrl that, depending on the version
indicated, a specific behavior is to be expected. This allows us to easily
support multiple versions of gym without requiring any effort from the user side.
For example, considering that our virtual environment has the v0.26.2 installed,
the following function will return `1` when queried:

```
>>> from torchrl._utils import implement_for
>>> @implement_for("gym", None, "0.26.0")
... def fun():
... return 0
>>> @implement_for("gym", "0.26.0", None)
... def fun():
... return 1
>>> fun()
1
```

## Available wrappers

| [`BraxEnv`](generated/torchrl.envs.BraxEnv.html#torchrl.envs.BraxEnv)(*args[, num_workers]) | Google Brax environment wrapper built with the environment name. |
| --- | --- |
| [`BraxWrapper`](generated/torchrl.envs.BraxWrapper.html#torchrl.envs.BraxWrapper)(*args, **kwargs) | Google Brax environment wrapper. |
| [`DMControlEnv`](generated/torchrl.envs.DMControlEnv.html#torchrl.envs.DMControlEnv)(*args[, num_workers]) | DeepMind Control lab environment wrapper. |
| [`DMControlWrapper`](generated/torchrl.envs.DMControlWrapper.html#torchrl.envs.DMControlWrapper)(*args, **kwargs) | DeepMind Control lab environment wrapper. |
| [`GenesisEnv`](generated/torchrl.envs.GenesisEnv.html#torchrl.envs.GenesisEnv)(*args[, num_workers]) | Genesis environment built from a named configuration. |
| [`GenesisWrapper`](generated/torchrl.envs.GenesisWrapper.html#torchrl.envs.GenesisWrapper)(*args, **kwargs) | TorchRL wrapper around a Genesis physics scene. |
| [`GymEnv`](generated/torchrl.envs.GymEnv.html#torchrl.envs.GymEnv)(*args, **kwargs) | OpenAI Gym environment wrapper constructed by environment ID directly. |
| [`GymWrapper`](generated/torchrl.envs.GymWrapper.html#torchrl.envs.GymWrapper)(*args, **kwargs) | OpenAI Gym environment wrapper. |
| [`HabitatEnv`](generated/torchrl.envs.HabitatEnv.html#torchrl.envs.HabitatEnv)(*args[, num_workers]) | A wrapper for habitat envs. |
| [`IsaacGymEnv`](generated/torchrl.envs.IsaacGymEnv.html#torchrl.envs.IsaacGymEnv)(*args, **kwargs) | A TorchRL Env interface for IsaacGym environments. |
| [`IsaacGymWrapper`](generated/torchrl.envs.IsaacGymWrapper.html#torchrl.envs.IsaacGymWrapper)(*args, **kwargs) | Wrapper for IsaacGymEnvs environments. |
| [`IsaacLabWrapper`](generated/torchrl.envs.IsaacLabWrapper.html#torchrl.envs.IsaacLabWrapper)(*args, **kwargs) | A wrapper for IsaacLab environments. |
| [`JumanjiEnv`](generated/torchrl.envs.JumanjiEnv.html#torchrl.envs.JumanjiEnv)(*args, **kwargs) | Jumanji environment wrapper built with the environment name. |
| [`JumanjiWrapper`](generated/torchrl.envs.JumanjiWrapper.html#torchrl.envs.JumanjiWrapper)(*args, **kwargs) | Jumanji's environment wrapper. |
| [`LiberoEnv`](generated/torchrl.envs.LiberoEnv.html#torchrl.envs.LiberoEnv)(*args[, num_workers, num_envs]) | LIBERO environment built from a task-suite name and task id. |
| [`LiberoWrapper`](generated/torchrl.envs.LiberoWrapper.html#torchrl.envs.LiberoWrapper)(*args, **kwargs) | LIBERO environment wrapper. |
| [`MeltingpotEnv`](generated/torchrl.envs.MeltingpotEnv.html#torchrl.envs.MeltingpotEnv)(*args, **kwargs) | Meltingpot environment wrapper. |
| [`MeltingpotWrapper`](generated/torchrl.envs.MeltingpotWrapper.html#torchrl.envs.MeltingpotWrapper)(*args, **kwargs) | Meltingpot environment wrapper. |
| [`MOGymEnv`](generated/torchrl.envs.MOGymEnv.html#torchrl.envs.MOGymEnv)(*args, **kwargs) | FARAMA MO-Gymnasium environment wrapper. |
| [`MOGymWrapper`](generated/torchrl.envs.MOGymWrapper.html#torchrl.envs.MOGymWrapper)(*args, **kwargs) | FARAMA MO-Gymnasium environment wrapper. |
| [`MJLabEnv`](generated/torchrl.envs.MJLabEnv.html#torchrl.envs.MJLabEnv)(*args[, num_workers]) | Build and wrap an mjlab task from mjlab's task registry. |
| [`MJLabWrapper`](generated/torchrl.envs.MJLabWrapper.html#torchrl.envs.MJLabWrapper)(*args, **kwargs) | TorchRL wrapper for a pre-built `mjlab.envs.ManagerBasedRlEnv`. |
| [`MujocoPlaygroundEnv`](generated/torchrl.envs.MujocoPlaygroundEnv.html#torchrl.envs.MujocoPlaygroundEnv)(*args[, num_workers]) | Google DeepMind MuJoCo Playground environment wrapper built with the environment name. |
| [`MujocoPlaygroundWrapper`](generated/torchrl.envs.MujocoPlaygroundWrapper.html#torchrl.envs.MujocoPlaygroundWrapper)(*args, **kwargs) | Google DeepMind MuJoCo Playground environment wrapper. |
| [`MujocoPlaygroundAgentMapping`](generated/torchrl.envs.MujocoPlaygroundAgentMapping.html#torchrl.envs.MujocoPlaygroundAgentMapping)(agents, ...) | Agent mapping for [`MujocoPlaygroundWrapper`](generated/torchrl.envs.MujocoPlaygroundWrapper.html#torchrl.envs.MujocoPlaygroundWrapper). |
| [`MujocoPlaygroundAgentSpec`](generated/torchrl.envs.MujocoPlaygroundAgentSpec.html#torchrl.envs.MujocoPlaygroundAgentSpec)(name, ...) | Observation/action slice definition for one agent in a cooperative task. |
| [`MultiThreadedEnv`](generated/torchrl.envs.MultiThreadedEnv.html#torchrl.envs.MultiThreadedEnv)(*args, **kwargs) | Multithreaded execution of environments based on EnvPool. |
| [`MultiThreadedEnvWrapper`](generated/torchrl.envs.MultiThreadedEnvWrapper.html#torchrl.envs.MultiThreadedEnvWrapper)(*args, **kwargs) | Wrapper for envpool-based multithreaded environments. |
| [`OpenMLEnv`](generated/torchrl.envs.OpenMLEnv.html#torchrl.envs.OpenMLEnv)(*args, **kwargs) | An environment interface to OpenML data to be used in bandits contexts. |
| [`OpenSpielWrapper`](generated/torchrl.envs.OpenSpielWrapper.html#torchrl.envs.OpenSpielWrapper)(*args, **kwargs) | Google DeepMind OpenSpiel environment wrapper. |
| [`OpenSpielEnv`](generated/torchrl.envs.OpenSpielEnv.html#torchrl.envs.OpenSpielEnv)(*args, **kwargs) | Google DeepMind OpenSpiel environment wrapper built with the game string. |
| [`PettingZooEnv`](generated/torchrl.envs.PettingZooEnv.html#torchrl.envs.PettingZooEnv)(*args, **kwargs) | PettingZoo Environment. |
| [`PettingZooWrapper`](generated/torchrl.envs.PettingZooWrapper.html#torchrl.envs.PettingZooWrapper)(*args, **kwargs) | PettingZoo environment wrapper. |
| [`ProcgenWrapper`](generated/torchrl.envs.ProcgenWrapper.html#torchrl.envs.ProcgenWrapper)(*args, **kwargs) | OpenAI Procgen environment wrapper. |
| [`RoboHiveEnv`](generated/torchrl.envs.RoboHiveEnv.html#torchrl.envs.RoboHiveEnv)(*args, **kwargs) | A wrapper for RoboHive gym environments. |
| [`SMACv2Env`](generated/torchrl.envs.SMACv2Env.html#torchrl.envs.SMACv2Env)(*args, **kwargs) | SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper. |
| [`SMACv2Wrapper`](generated/torchrl.envs.SMACv2Wrapper.html#torchrl.envs.SMACv2Wrapper)(*args, **kwargs) | SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper. |
| [`UnityMLAgentsEnv`](generated/torchrl.envs.UnityMLAgentsEnv.html#torchrl.envs.UnityMLAgentsEnv)(*args, **kwargs) | Unity ML-Agents environment wrapper. |
| [`UnityMLAgentsWrapper`](generated/torchrl.envs.UnityMLAgentsWrapper.html#torchrl.envs.UnityMLAgentsWrapper)(*args, **kwargs) | Unity ML-Agents environment wrapper. |
| [`VmasEnv`](generated/torchrl.envs.VmasEnv.html#torchrl.envs.VmasEnv)(*args, **kwargs) | Vmas environment wrapper. |
| [`VmasWrapper`](generated/torchrl.envs.VmasWrapper.html#torchrl.envs.VmasWrapper)(*args, **kwargs) | Vmas environment wrapper. |
| [`gym_backend`](generated/torchrl.envs.gym_backend.html#torchrl.envs.gym_backend)([submodule]) | Returns the gym backend, or a submodule of it. |
| [`set_gym_backend`](generated/torchrl.envs.set_gym_backend.html#torchrl.envs.set_gym_backend)(backend) | Sets the gym-backend to a certain value. |
| [`register_gym_spec_conversion`](generated/torchrl.envs.register_gym_spec_conversion.html#torchrl.envs.register_gym_spec_conversion)(spec_type) | Decorator to register a conversion function for a specific spec type. |

## Auto-resetting Environments

Auto-resetting environments are environments where calls to [`reset()`](generated/torchrl.envs.EnvBase.html#id1) are not expected when
the environment reaches a `"done"` state during a rollout, as the reset happens automatically.
Usually, in such cases the observations delivered with the done and reward (which effectively result from performing the
action in the environment) are actually the first observations of a new episode, and not the last observations of the
current episode.

To handle these cases, torchrl provides a [`AutoResetTransform`](generated/torchrl.envs.transforms.AutoResetTransform.html#torchrl.envs.transforms.AutoResetTransform) that will copy the observations
that result from the call to step to the next reset and skip the calls to reset during rollouts (in both
[`rollout()`](generated/torchrl.envs.EnvBase.html#id2) and [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) iterations).
This transform class also provides a fine-grained control over the behavior to be adopted for the invalid observations,
which can be masked with "nan" or any other values, or not masked at all.

To tell torchrl that an environment is auto-resetting, it is sufficient to provide an `auto_reset` argument
during construction. If provided, an `auto_reset_replace` argument can also control whether the values of the last
observation of an episode should be replaced with some placeholder or not.

```
>>> from torchrl.envs import GymEnv
>>> from torchrl.envs import set_gym_backend
>>> import torch
>>> torch.manual_seed(0)
>>>
>>> class AutoResettingGymEnv(GymEnv):
... def _step(self, tensordict):
... tensordict = super()._step(tensordict)
... if tensordict["done"].any():
... td_reset = super().reset()
... tensordict.update(td_reset.exclude(*self.done_keys))
... return tensordict
...
... def _reset(self, tensordict=None):
... if tensordict is not None and "_reset" in tensordict:
... return tensordict.copy()
... return super()._reset(tensordict)
>>>
>>> with set_gym_backend("gym"):
... env = AutoResettingGymEnv("CartPole-v1", auto_reset=True, auto_reset_replace=True)
... env.set_seed(0)
... r = env.rollout(30, break_when_any_done=False)
>>> print(r["next", "done"].squeeze())
tensor([False, False, False, False, False, False, False, False, False, False,
 False, False, False, True, False, False, False, False, False, False,
 False, False, False, False, False, True, False, False, False, False])
```

## Dynamic Specs

Running environments in parallel is usually done via the creation of memory buffers used to pass information from one
process to another. In some cases, it may be impossible to forecast whether an environment will or will not have
consistent inputs or outputs during a rollout, as their shape may be variable. We refer to this as dynamic specs.

TorchRL is capable of handling dynamic specs, but the batched environments and collectors will need to be made
aware of this feature. Note that, in practice, this is detected automatically.

To indicate that a tensor will have a variable size along a dimension, one can set the size value as `-1` for the
desired dimensions. Because the data cannot be stacked contiguously, calls to `env.rollout` need to be made with
the `return_contiguous=False` argument.
Here is a working example:

```
>>> from torchrl.envs import EnvBase
>>> from torchrl.data import Unbounded, Composite, Bounded, Binary
>>> import torch
>>> from tensordict import TensorDict, TensorDictBase
>>>
>>> class EnvWithDynamicSpec(EnvBase):
... def __init__(self, max_count=5):
... super().__init__(batch_size=())
... self.observation_spec = Composite(
... observation=Unbounded(shape=(3, -1, 2)),
... )
... self.action_spec = Bounded(low=-1, high=1, shape=(2,))
... self.full_done_spec = Composite(
... done=Binary(1, shape=(1,), dtype=torch.bool),
... terminated=Binary(1, shape=(1,), dtype=torch.bool),
... truncated=Binary(1, shape=(1,), dtype=torch.bool),
... )
... self.reward_spec = Unbounded((1,), dtype=torch.float)
... self.count = 0
... self.max_count = max_count
...
... def _reset(self, tensordict=None):
... self.count = 0
... data = TensorDict(
... {
... "observation": torch.full(
... (3, self.count + 1, 2),
... self.count,
... dtype=self.observation_spec["observation"].dtype,
... )
... }
... )
... data.update(self.done_spec.zero())
... return data
...
... def _step(
... self,
... tensordict: TensorDictBase,
... ) -> TensorDictBase:
... self.count += 1
... done = self.count >= self.max_count
... observation = TensorDict(
... {
... "observation": torch.full(
... (3, self.count + 1, 2),
... self.count,
... dtype=self.observation_spec["observation"].dtype,
... )
... }
... )
... done = self.full_done_spec.zero() | done
... reward = self.full_reward_spec.zero()
... return observation.update(done).update(reward)
...
... def _set_seed(self, seed: Optional[int]) -> None:
... self.manual_seed = seed
... return seed
>>> env = EnvWithDynamicSpec()
>>> print(env.rollout(5, return_contiguous=False))
LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: LazyStackedTensorDict(
 fields={
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 3, -1, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([5]),
 device=None,
 is_shared=False,
 stack_dim=0),
 observation: Tensor(shape=torch.Size([5, 3, -1, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([5]),
 device=None,
 is_shared=False,
 stack_dim=0)
```

Warning

The absence of memory buffers in [`ParallelEnv`](generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) and in
data collectors can impact performance of these classes dramatically. Any
such usage should be carefully benchmarked against a plain execution on a
single process, as serializing and deserializing large numbers of tensors
can be very expensive.

Currently, [`check_env_specs()`](generated/torchrl.envs.check_env_specs.html#torchrl.envs.check_env_specs) will pass for dynamic specs where a shape varies along some
dimensions, but not when a key is present during a step and absent during others, or when the number of dimensions
varies.