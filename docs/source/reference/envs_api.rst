.. currentmodule:: torchrl.envs

.. _Environment-API:

Environment API
===============

TorchRL offers an API to handle environments of different backends, such as gym,
dm-control, dm-lab, model-based environments as well as custom environments.
The goal is to be able to swap environments in an experiment with little or no effort,
even if these environments are simulated using different libraries.
TorchRL offers some out-of-the-box environment wrappers under :mod:`torchrl.envs.libs`,
which we hope can be easily imitated for other libraries.
The parent class :class:`~torchrl.envs.EnvBase` is a :class:`torch.nn.Module` subclass that implements
some typical environment methods using :class:`tensordict.TensorDict` as a data organiser. This allows this
class to be generic and to handle an arbitrary number of input and outputs, as well as
nested or batched data structures.

Each env will have the following attributes:

- :attr:`env.batch_size`: a :class:`torch.Size` representing the number of envs
  batched together.
- :attr:`env.device`: the device where the input and output tensordict are expected to live.
  The environment device does not mean that the actual step operations will be computed on device
  (this is the responsibility of the backend, with which TorchRL can do little). The device of
  an environment just represents the device where the data is to be expected when input to the
  environment or retrieved from it. TorchRL takes care of mapping the data to the desired device.
  This is especially useful for transforms (see below). For parametric environments (e.g.
  model-based environments), the device does represent the hardware that will be used to
  compute the operations.
- :attr:`env.observation_spec`: a :class:`~torchrl.data.Composite` object
  containing all the observation key-spec pairs.
- :attr:`env.state_spec`: a :class:`~torchrl.data.Composite` object
  containing all the input key-spec pairs (except action). For most stateful
  environments, this container will be empty.
- :attr:`env.action_spec`: a :class:`~torchrl.data.TensorSpec` object
  representing the action spec.
- :attr:`env.reward_spec`: a :class:`~torchrl.data.TensorSpec` object representing
  the reward spec.
- :attr:`env.done_spec`: a :class:`~torchrl.data.TensorSpec` object representing
  the done-flag spec. See the section on trajectory termination below.
- :attr:`env.input_spec`: a :class:`~torchrl.data.Composite` object containing
  all the input keys (``"full_action_spec"`` and ``"full_state_spec"``).
- :attr:`env.output_spec`: a :class:`~torchrl.data.Composite` object containing
  all the output keys (``"full_observation_spec"``, ``"full_reward_spec"`` and ``"full_done_spec"``).

If the environment carries non-tensor data, a :class:`~torchrl.data.NonTensor`
instance can be used.

Env specs: locks and batch size
-------------------------------

.. _Environment-lock:

Environment specs are locked by default (through a ``spec_locked`` arg passed to the env constructor).
Locking specs means that any modification of the spec (or its children if it is a :class:`~torchrl.data.Composite`
instance) will require to unlock it. This can be done via the :meth:`~torchrl.envs.EnvBase.set_spec_lock_`.
The reason specs are locked by default is that it makes it easy to cache values such as action or reset keys and the
likes.
Unlocking an env should only be done if it expected that the specs will be modified often (which, in principle, should
be avoided).
Modifications of the specs such as `env.observation_spec = new_spec` are allowed: under the hood, TorchRL will erase
the cache, unlock the specs, make the modification and relock the specs if the env was previously locked.

Importantly, the environment spec shapes should contain the batch size, e.g.
an environment with :attr:`env.batch_size` ``== torch.Size([4])`` should have
an :attr:`env.action_spec` with shape :class:`torch.Size` ``([4, action_size])``.
This is helpful when preallocation tensors, checking shape consistency etc.

Env methods
-----------

With these, the following methods are implemented:

- :meth:`env.reset`: a reset method that may (but not necessarily requires to) take
  a :class:`tensordict.TensorDict` input. It return the first tensordict of a rollout, usually
  containing a ``"done"`` state and a set of observations. If not present,
  a ``"reward"`` key will be instantiated with 0s and the appropriate shape.
- :meth:`env.step`: a step method that takes a :class:`tensordict.TensorDict` input
  containing an input action as well as other inputs (for model-based or stateless
  environments, for instance).
- :meth:`env.step_and_maybe_reset`: executes a step, and (partially) resets the
  environments if it needs to. It returns the updated input with a ``"next"``
  key containing the data of the next step, as well as a tensordict containing
  the input data for the next step (ie, reset or result or
  :func:`~torchrl.envs.utils.step_mdp`)
  This is done by reading the ``done_keys`` and
  assigning a ``"_reset"`` signal to each done state. This method allows
  to code non-stopping rollout functions with little effort:

    >>> data_ = env.reset()
    >>> result = []
    >>> for i in range(N):
    ...     data, data_ = env.step_and_maybe_reset(data_)
    ...     result.append(data)
    ...
    >>> result = torch.stack(result)

- :meth:`env.set_seed`: a seeding method that will return the next seed
  to be used in a multi-env setting. This next seed is deterministically computed
  from the preceding one, such that one can seed multiple environments with a different
  seed without risking to overlap seeds in consecutive experiments, while still
  having reproducible results.
- :meth:`env.rollout`: executes a rollout in the environment for
  a maximum number of steps (``max_steps=N``) and using a policy (``policy=model``).
  The policy should be coded using a :class:`tensordict.nn.TensorDictModule`
  (or any other :class:`tensordict.TensorDict`-compatible module).
  The resulting :class:`tensordict.TensorDict` instance will be marked with
  a trailing ``"time"`` named dimension that can be used by other modules
  to treat this batched dimension as it should.

The following figure summarizes how a rollout is executed in torchrl.

.. figure:: /_static/img/rollout.gif

   TorchRL rollouts using TensorDict.

In brief, a TensorDict is created by the :meth:`~.EnvBase.reset` method,
then populated with an action by the policy before being passed to the
:meth:`~.EnvBase.step` method which writes the observations, done flag(s) and
reward under the ``"next"`` entry. The result of this call is stored for
delivery and the ``"next"`` entry is gathered by the :func:`~.utils.step_mdp`
function.

.. note::
  In general, all TorchRL environment have a ``"done"`` and ``"terminated"``
  entry in their output tensordict. If they are not present by design,
  the :class:`~.EnvBase` metaclass will ensure that every done or terminated
  is flanked with its dual.
  In TorchRL, ``"done"`` strictly refers to the union of all the end-of-trajectory
  signals and should be interpreted as "the last step of a trajectory" or
  equivalently "a signal indicating the need to reset".
  If the environment provides it (eg, Gymnasium), the truncation entry is also
  written in the :meth:`EnvBase.step` output under a ``"truncated"`` entry.
  If the environment carries a single value, it will interpreted as a ``"terminated"``
  signal by default.
  By default, TorchRL's collectors and rollout methods will be looking for the ``"done"``
  entry to assess if the environment should be reset.

.. note::

  The `torchrl.collectors.utils.split_trajectories` function can be used to
  slice adjacent trajectories. It relies on a ``"traj_ids"`` entry in the
  input tensordict, or to the junction of ``"done"`` and ``"truncated"`` key
  if the ``"traj_ids"`` is missing.


.. note::

  In some contexts, it can be useful to mark the first step of a trajectory.
  TorchRL provides such functionality through the :class:`~torchrl.envs.InitTracker`
  transform.


Our environment :ref:`tutorial <pendulum_tuto>`
provides more information on how to design a custom environment from scratch.

Base classes
------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    EnvBase
    GymLikeEnv
    EnvMetaData

Custom native TorchRL environments
----------------------------------

TorchRL offers a series of custom built-in environments.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChessEnv
    FinancialRegimeEnv
    LLMHashingEnv
    PendulumEnv
    TicTacToeEnv

Domain-specific
---------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    ModelBasedEnvBase
    model_based.dreamer.DreamerEnv
    model_based.dreamer.DreamerDecoder

Helpers
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    check_env_specs
    exploration_type
    get_available_libraries
    make_composite_from_td
    set_exploration_type
    step_mdp
    terminated_or_truncated
