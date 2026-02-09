.. currentmodule:: torchrl.envs

Library Wrappers
================

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
:class:`~.gym.set_gym_backend` allows to control which library will be used
in the relevant functions:

    >>> from torchrl.envs.libs.gym import GymEnv, set_gym_backend, gym_backend
    >>> import gymnasium, gym
    >>> with set_gym_backend(gymnasium):
    ...     print(gym_backend())
    ...     env1 = GymEnv("Pendulum-v1")
    <module 'gymnasium' from '/path/to/venv/python3.10/site-packages/gymnasium/__init__.py'>
    >>> with set_gym_backend(gym):
    ...     print(gym_backend())
    ...     env2 = GymEnv("Pendulum-v1")
    <module 'gym' from '/path/to/venv/python3.10/site-packages/gym/__init__.py'>
    >>> print(env1._env.env.env)
    <gymnasium.envs.classic_control.pendulum.PendulumEnv at 0x15147e190>
    >>> print(env2._env.env.env)
    <gym.envs.classic_control.pendulum.PendulumEnv at 0x1629916a0>

We can see that the two libraries modify the value returned by :func:`~torchrl.envs.gym.gym_backend()`
which can be further used to indicate which library needs to be used for
the current computation. :class:`~.gym.set_gym_backend` is also a decorator:
we can use it to tell to a specific function what gym backend needs to be used
during its execution.
The :func:`torchrl.envs.libs.gym.gym_backend` function allows you to gather
the current gym backend or any of its modules:

        >>> import mo_gymnasium
        >>> with set_gym_backend("gym"):
        ...     wrappers = gym_backend('wrappers')
        ...     print(wrappers)
        <module 'gym.wrappers' from '/path/to/venv/python3.10/site-packages/gym/wrappers/__init__.py'>
        >>> with set_gym_backend("gymnasium"):
        ...     wrappers = gym_backend('wrappers')
        ...     print(wrappers)
        <module 'gymnasium.wrappers' from '/path/to/venv/python3.10/site-packages/gymnasium/wrappers/__init__.py'>

Another tool that comes in handy with gym and other external dependencies is
the :class:`torchrl._utils.implement_for` class. Decorating a function
with ``@implement_for`` will tell torchrl that, depending on the version
indicated, a specific behavior is to be expected. This allows us to easily
support multiple versions of gym without requiring any effort from the user side.
For example, considering that our virtual environment has the v0.26.2 installed,
the following function will return ``1`` when queried:

    >>> from torchrl._utils import implement_for
    >>> @implement_for("gym", None, "0.26.0")
    ... def fun():
    ...     return 0
    >>> @implement_for("gym", "0.26.0", None)
    ... def fun():
    ...     return 1
    >>> fun()
    1

Available wrappers
------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    BraxEnv
    BraxWrapper
    DMControlEnv
    DMControlWrapper
    GymEnv
    GymWrapper
    HabitatEnv
    IsaacGymEnv
    IsaacGymWrapper
    IsaacLabWrapper
    JumanjiEnv
    JumanjiWrapper
    MeltingpotEnv
    MeltingpotWrapper
    MOGymEnv
    MOGymWrapper
    MultiThreadedEnv
    MultiThreadedEnvWrapper
    OpenEnvEnv
    OpenEnvWrapper
    OpenMLEnv
    OpenSpielWrapper
    OpenSpielEnv
    PettingZooEnv
    PettingZooWrapper
    ProcgenWrapper
    RoboHiveEnv
    SMACv2Env
    SMACv2Wrapper
    UnityMLAgentsEnv
    UnityMLAgentsWrapper
    VmasEnv
    VmasWrapper
    gym_backend
    set_gym_backend
    register_gym_spec_conversion

Auto-resetting Environments
---------------------------

.. _autoresetting_envs:

Auto-resetting environments are environments where calls to :meth:`~torchrl.envs.EnvBase.reset` are not expected when
the environment reaches a ``"done"`` state during a rollout, as the reset happens automatically.
Usually, in such cases the observations delivered with the done and reward (which effectively result from performing the
action in the environment) are actually the first observations of a new episode, and not the last observations of the
current episode.

To handle these cases, torchrl provides a :class:`~torchrl.envs.AutoResetTransform` that will copy the observations
that result from the call to `step` to the next `reset` and skip the calls to `reset` during rollouts (in both
:meth:`~torchrl.envs.EnvBase.rollout` and :class:`~torchrl.collectors.SyncDataCollector` iterations).
This transform class also provides a fine-grained control over the behavior to be adopted for the invalid observations,
which can be masked with `"nan"` or any other values, or not masked at all.

To tell torchrl that an environment is auto-resetting, it is sufficient to provide an ``auto_reset`` argument
during construction. If provided, an ``auto_reset_replace`` argument can also control whether the values of the last
observation of an episode should be replaced with some placeholder or not.

  >>> from torchrl.envs import GymEnv
  >>> from torchrl.envs import set_gym_backend
  >>> import torch
  >>> torch.manual_seed(0)
  >>>
  >>> class AutoResettingGymEnv(GymEnv):
  ...     def _step(self, tensordict):
  ...         tensordict = super()._step(tensordict)
  ...         if tensordict["done"].any():
  ...             td_reset = super().reset()
  ...             tensordict.update(td_reset.exclude(*self.done_keys))
  ...         return tensordict
  ...
  ...     def _reset(self, tensordict=None):
  ...         if tensordict is not None and "_reset" in tensordict:
  ...             return tensordict.copy()
  ...         return super()._reset(tensordict)
  >>>
  >>> with set_gym_backend("gym"):
  ...     env = AutoResettingGymEnv("CartPole-v1", auto_reset=True, auto_reset_replace=True)
  ...     env.set_seed(0)
  ...     r = env.rollout(30, break_when_any_done=False)
  >>> print(r["next", "done"].squeeze())
  tensor([False, False, False, False, False, False, False, False, False, False,
          False, False, False,  True, False, False, False, False, False, False,
          False, False, False, False, False,  True, False, False, False, False])

Dynamic Specs
-------------

.. _dynamic_envs:

Running environments in parallel is usually done via the creation of memory buffers used to pass information from one
process to another. In some cases, it may be impossible to forecast whether an environment will or will not have
consistent inputs or outputs during a rollout, as their shape may be variable. We refer to this as dynamic specs.

TorchRL is capable of handling dynamic specs, but the batched environments and collectors will need to be made
aware of this feature. Note that, in practice, this is detected automatically.

To indicate that a tensor will have a variable size along a dimension, one can set the size value as ``-1`` for the
desired dimensions. Because the data cannot be stacked contiguously, calls to ``env.rollout`` need to be made with
the ``return_contiguous=False`` argument.
Here is a working example:

    >>> from torchrl.envs import EnvBase
    >>> from torchrl.data import Unbounded, Composite, Bounded, Binary
    >>> import torch
    >>> from tensordict import TensorDict, TensorDictBase
    >>>
    >>> class EnvWithDynamicSpec(EnvBase):
    ...     def __init__(self, max_count=5):
    ...         super().__init__(batch_size=())
    ...         self.observation_spec = Composite(
    ...             observation=Unbounded(shape=(3, -1, 2)),
    ...         )
    ...         self.action_spec = Bounded(low=-1, high=1, shape=(2,))
    ...         self.full_done_spec = Composite(
    ...             done=Binary(1, shape=(1,), dtype=torch.bool),
    ...             terminated=Binary(1, shape=(1,), dtype=torch.bool),
    ...             truncated=Binary(1, shape=(1,), dtype=torch.bool),
    ...         )
    ...         self.reward_spec = Unbounded((1,), dtype=torch.float)
    ...         self.count = 0
    ...         self.max_count = max_count
    ...
    ...     def _reset(self, tensordict=None):
    ...         self.count = 0
    ...         data = TensorDict(
    ...             {
    ...                 "observation": torch.full(
    ...                     (3, self.count + 1, 2),
    ...                     self.count,
    ...                     dtype=self.observation_spec["observation"].dtype,
    ...                 )
    ...             }
    ...         )
    ...         data.update(self.done_spec.zero())
    ...         return data
    ...
    ...     def _step(
    ...         self,
    ...         tensordict: TensorDictBase,
    ...     ) -> TensorDictBase:
    ...         self.count += 1
    ...         done = self.count >= self.max_count
    ...         observation = TensorDict(
    ...             {
    ...                 "observation": torch.full(
    ...                     (3, self.count + 1, 2),
    ...                     self.count,
    ...                     dtype=self.observation_spec["observation"].dtype,
    ...                 )
    ...             }
    ...         )
    ...         done = self.full_done_spec.zero() | done
    ...         reward = self.full_reward_spec.zero()
    ...         return observation.update(done).update(reward)
    ...
    ...     def _set_seed(self, seed: Optional[int]) -> None:
    ...         self.manual_seed = seed
    ...         return seed
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

.. warning::
  The absence of memory buffers in :class:`~torchrl.envs.ParallelEnv` and in
  data collectors can impact performance of these classes dramatically. Any
  such usage should be carefully benchmarked against a plain execution on a
  single process, as serializing and deserializing large numbers of tensors
  can be very expensive.

Currently, :func:`~torchrl.envs.utils.check_env_specs` will pass for dynamic specs where a shape varies along some
dimensions, but not when a key is present during a step and absent during others, or when the number of dimensions
varies.
