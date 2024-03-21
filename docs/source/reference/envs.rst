.. currentmodule:: torchrl.envs

torchrl.envs package
====================

.. _Environment-API:

TorchRL offers an API to handle environments of different backends, such as gym,
dm-control, dm-lab, model-based environments as well as custom environments.
The goal is to be able to swap environments in an experiment with little or no effort,
even if these environments are simulated using different libraries.
TorchRL offers some out-of-the-box environment wrappers under :obj:`torchrl.envs.libs`,
which we hope can be easily imitated for other libraries.
The parent class :class:`~torchrl.envs.EnvBase` is a :class:`torch.nn.Module` subclass that implements
some typical environment methods using :class:`tensordict.TensorDict` as a data organiser. This allows this
class to be generic and to handle an arbitrary number of input and outputs, as well as
nested or batched data structures.

Each env will have the following attributes:

- :obj:`env.batch_size`: a :obj:`torch.Size` representing the number of envs
  batched together.
- :obj:`env.device`: the device where the input and output tensordict are expected to live.
  The environment device does not mean that the actual step operations will be computed on device
  (this is the responsibility of the backend, with which TorchRL can do little). The device of
  an environment just represents the device where the data is to be expected when input to the
  environment or retrieved from it. TorchRL takes care of mapping the data to the desired device.
  This is especially useful for transforms (see below). For parametric environments (e.g.
  model-based environments), the device does represent the hardware that will be used to
  compute the operations.
- :obj:`env.observation_spec`: a :class:`~torchrl.data.CompositeSpec` object
  containing all the observation key-spec pairs.
- :obj:`env.state_spec`: a :class:`~torchrl.data.CompositeSpec` object
  containing all the input key-spec pairs (except action). For most stateful
  environments, this container will be empty.
- :obj:`env.action_spec`: a :class:`~torchrl.data.TensorSpec` object
  representing the action spec.
- :obj:`env.reward_spec`: a :class:`~torchrl.data.TensorSpec` object representing
  the reward spec.
- :obj:`env.done_spec`: a :class:`~torchrl.data.TensorSpec` object representing
  the done-flag spec. See the section on trajectory termination below.
- :obj:`env.input_spec`: a :class:`~torchrl.data.CompositeSpec` object containing
  all the input keys (:obj:`"full_action_spec"` and :obj:`"full_state_spec"`).
  It is locked and should not be modified directly.
- :obj:`env.output_spec`: a :class:`~torchrl.data.CompositeSpec` object containing
  all the output keys (:obj:`"full_observation_spec"`, :obj:`"full_reward_spec"` and :obj:`"full_done_spec"`).
  It is locked and should not be modified directly.

Importantly, the environment spec shapes should contain the batch size, e.g.
an environment with :obj:`env.batch_size == torch.Size([4])` should have
an :obj:`env.action_spec` with shape :obj:`torch.Size([4, action_size])`.
This is helpful when preallocation tensors, checking shape consistency etc.

With these, the following methods are implemented:

- :meth:`env.reset`: a reset method that may (but not necessarily requires to) take
  a :class:`tensordict.TensorDict` input. It return the first tensordict of a rollout, usually
  containing a :obj:`"done"` state and a set of observations. If not present,
  a `"reward"` key will be instantiated with 0s and the appropriate shape.
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


Our environment `tutorial <https://pytorch.org/rl/tutorials/pendulum.html>`_
provides more information on how to design a custom environment from scratch.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    EnvBase
    GymLikeEnv
    EnvMetaData

Vectorized envs
---------------

Vectorized (or better: parallel) environments is a common feature in Reinforcement Learning
where executing the environment step can be cpu-intensive.
Some libraries such as `gym3 <https://github.com/openai/gym3>`_ or `EnvPool <https://github.com/sail-sg/envpool>`_
offer interfaces to execute batches of environments simultaneously.
While they often offer a very competitive computational advantage, they do not
necessarily scale to the wide variety of environment libraries supported by TorchRL.
Therefore, TorchRL offers its own, generic :class:`ParallelEnv` class to run multiple
environments in parallel.
As this class inherits from :class:`SerialEnv`, it enjoys the exact same API as other environment.
Of course, a :class:`ParallelEnv` will have a batch size that corresponds to its environment count:

.. note::
  Given the library's many optional dependencies (eg, Gym, Gymnasium, and many others)
  warnings can quickly become quite annoying in multiprocessed / distributed settings.
  By default, TorchRL filters out these warnings in sub-processes. If one still wishes to
  see these warnings, they can be displayed by setting ``torchrl.filter_warnings_subprocess=False``.

It is important that your environment specs match the input and output that it sends and receives, as
:class:`ParallelEnv` will create buffers from these specs to communicate with the spawn processes.
Check the :func:`~torchrl.envs.utils.check_env_specs` method for a sanity check.

.. code-block::
   :caption: Parallel environment

        >>> def make_env():
        ...     return GymEnv("Pendulum-v1", from_pixels=True, g=9.81, device="cuda:0")
        >>> check_env_specs(env)  # this must pass for ParallelEnv to work
        >>> env = ParallelEnv(4, make_env)
        >>> print(env.batch_size)
        torch.Size([4])

:class:`ParallelEnv` allows to retrieve the attributes from its contained environments:
one can simply call:

.. code-block::
   :caption: Parallel environment attributes

        >>> a, b, c, d = env.g  # gets the g-force of the various envs, which we set to 9.81 before
        >>> print(a)
        9.81

TorchRL uses a private ``"_reset"`` key to indicate to the environment which
component (sub-environments or agents) should be reset.
This allows to reset some but not all of the components.

The ``"_reset"`` key has two distinct functionalities:
1. During a call to :meth:`~.EnvBase._reset`, the ``"_reset"`` key may or may
   not be present in the input tensordict. TorchRL's convention is that the
   absence of the ``"_reset"`` key at a given ``"done"`` level indicates
   a total reset of that level (unless a ``"_reset"`` key was found at a level
   above, see details below).
   If it is present, it is expected that those entries and only those components
   where the ``"_reset"`` entry is ``True`` (along key and shape dimension) will be reset.

   The way an environment deals with the ``"_reset"`` keys in its :meth:`~.EnvBase._reset`
   method is proper to its class.
   Designing an environment that behaves according to ``"_reset"`` inputs is the
   developer's responsibility, as TorchRL has no control over the inner logic
   of :meth:`~.EnvBase._reset`. Nevertheless, the following point should be
   kept in mind when desiging that method.

2. After a call to :meth:`~.EnvBase._reset`, the output will be masked with the
   ``"_reset"`` entries and the output of the previous :meth:`~.EnvBase.step`
   will be written wherever the ``"_reset"`` was ``False``. In practice, this
   means that if a ``"_reset"`` modifies data that isn't exposed by it, this
   modification will be lost. After this masking operation, the ``"_reset"``
   entries will be erased from the :meth:`~.EnvBase.reset` outputs.

It must be pointed that ``"_reset"`` is a private key, and it should only be
used when coding specific environment features that are internal facing.
In other words, this should NOT be used outside of the library, and developers
will keep the right to modify the logic of partial resets through ``"_reset"``
setting without preliminary warranty, as long as they don't affect TorchRL
internal tests.

Finally, the following assumptions are made and should be kept in mind when
designing reset functionalities:

- Each ``"_reset"`` is paired with a ``"done"`` entry (+ ``"terminated"`` and,
  possibly, ``"truncated"``). This means that the following structure is not
  allowed: ``TensorDict({"done": done, "nested": {"_reset": reset}}, [])``, as
  the ``"_reset"`` lives at a different nesting level than the ``"done"``.
- A reset at one level does not preclude the presence of a ``"_reset"`` at lower
  levels, but it annihilates its effects. The reason is simply that
  whether the ``"_reset"`` at the root level corresponds to an ``all()``, ``any()``
  or custom call to the nested ``"done"`` entries cannot be known in advance,
  and it is explicitly assumed that the ``"_reset"`` at the root was placed
  there to superseed the nested values (for an example, have a look at
  :class:`~.PettingZooWrapper` implementation where each group has one or more
  ``"done"`` entries associated which is aggregated at the root level with a
  ``any`` or ``all`` logic depending on the task).
- When calling :meth:`env.reset(tensordict)` with a partial ``"_reset"`` entry
  that will reset some but not all the done sub-environments, the input data
  should contain the data of the sub-environemtns that are __not__ being reset.
  The reason for this constrain lies in the fact that the output of the
  ``env._reset(data)`` can only be predicted for the entries that are reset.
  For the others, TorchRL cannot know in advance if they will be meaningful or
  not. For instance, one could perfectly just pad the values of the non-reset
  components, in which case the non-reset data will be meaningless and should
  be discarded.

Below, we give some examples of the expected effect that ``"_reset"`` keys will
have on an environment returning zeros after reset:

    >>> # single reset at the root
    >>> data = TensorDict({"val": [1, 1], "_reset": [False, True]}, [])
    >>> env.reset(data)
    >>> print(data.get("val"))  # only the second value is 0
    tensor([1, 0])
    >>> # nested resets
    >>> data = TensorDict({
    ...     ("agent0", "val"): [1, 1], ("agent0", "_reset"): [False, True],
    ...     ("agent1", "val"): [2, 2], ("agent1", "_reset"): [True, False],
    ... }, [])
    >>> env.reset(data)
    >>> print(data.get(("agent0", "val")))  # only the second value is 0
    tensor([1, 0])
    >>> print(data.get(("agent1", "val")))  # only the second value is 0
    tensor([0, 2])
    >>> # nested resets are overridden by a "_reset" at the root
    >>> data = TensorDict({
    ...     "_reset": [True, True],
    ...     ("agent0", "val"): [1, 1], ("agent0", "_reset"): [False, True],
    ...     ("agent1", "val"): [2, 2], ("agent1", "_reset"): [True, False],
    ... }, [])
    >>> env.reset(data)
    >>> print(data.get(("agent0", "val")))  # reset at the root overrides nested
    tensor([0, 0])
    >>> print(data.get(("agent1", "val")))  # reset at the root overrides nested
    tensor([0, 0])

.. code-block::
   :caption: Parallel environment reset

        >>> tensordict = TensorDict({"_reset": [[True], [False], [True], [True]]}, [4])
        >>> env.reset(tensordict)  # eliminates the "_reset" entry
        TensorDict(
            fields={
                terminated: Tensor(torch.Size([4, 1]), dtype=torch.bool),
                done: Tensor(torch.Size([4, 1]), dtype=torch.bool),
                pixels: Tensor(torch.Size([4, 500, 500, 3]), dtype=torch.uint8),
                truncated: Tensor(torch.Size([4, 1]), dtype=torch.bool),
            batch_size=torch.Size([4]),
            device=None,
            is_shared=True)


.. note::

  *A note on performance*: launching a :class:`~.ParallelEnv` can take quite some time
  as it requires to launch as many python instances as there are processes. Due to
  the time that it takes to run ``import torch`` (and other imports), starting the
  parallel env can be a bottleneck. This is why, for instance, TorchRL tests are so slow.
  Once the environment is launched, a great speedup should be observed.

.. note::

  *TorchRL requires precise specs*: Another thing to take in consideration is
  that :class:`ParallelEnv` (as well as data collectors)
  will create data buffers based on the environment specs to pass data from one process
  to another. This means that a misspecified spec (input, observation or reward) will
  cause a breakage at runtime as the data can't be written on the preallocated buffer.
  In general, an environment should be tested using the :func:`~.utils.check_env_specs`
  test function before being used in a :class:`ParallelEnv`. This function will raise
  an assertion error whenever the preallocated buffer and the collected data mismatch.

We also offer the :class:`~.SerialEnv` class that enjoys the exact same API but is executed
serially. This is mostly useful for testing purposes, when one wants to assess the
behaviour of a :class:`~.ParallelEnv` without launching the subprocesses.

In addition to :class:`~.ParallelEnv`, which offers process-based parallelism, we also provide a way to create
multithreaded environments with :obj:`~.MultiThreadedEnv`. This class uses `EnvPool <https://github.com/sail-sg/envpool>`_
library underneath, which allows for higher performance, but at the same time restricts flexibility - one can only
create environments implemented in ``EnvPool``. This covers many popular RL environments types (Atari, Classic Control,
etc.), but one can not use an arbitrary TorchRL environment, as it is possible with :class:`~.ParallelEnv`. Run
`benchmarks/benchmark_batched_envs.py` to compare performance of different ways to parallelize batched environments.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    SerialEnv
    ParallelEnv
    EnvCreator

Multi-agent environments
------------------------

.. _MARL-environment-API:

.. currentmodule:: torchrl.envs

TorchRL supports multi-agent learning out-of-the-box.
*The same classes used in a single-agent learning pipeline can be seamlessly used in multi-agent contexts,
without any modification or dedicated multi-agent infrastructure.*

In this view, environments play a core role for multi-agent. In multi-agent environments,
many decision-making agents act in a shared world.
Agents can observe different things, act in different ways and also be rewarded differently.
Therefore, many paradigms exist to model multi-agent environments (DecPODPs, Markov Games).
Some of the main differences between these paradigms include:

- **observation** can be per-agent and also have some shared components
- **reward** can be per-agent or shared
- **done** (and ``"truncated"`` or ``"terminated"``) can be per-agent or shared.

TorchRL accommodates all these possible paradigms thanks to its :class:`tensordict.TensorDict` data carrier.
In particular, in multi-agent environments, per-agent keys will be carried in a nested "agents" TensorDict.
This TensorDict will have the additional agent dimension and thus group data that is different for each agent.
The shared keys, on the other hand, will be kept in the first level, as in single-agent cases.

Let's look at an example to understand this better. For this example we are going to use
`VMAS <https://github.com/proroklab/VectorizedMultiAgentSimulator>`_, a multi-robot task simulator also
based on PyTorch, which runs parallel batched simulation on device.

We can create a VMAS environment and look at what the output from a random step looks like:

.. code-block::
   :caption: Example of multi-agent step tensordict

        >>> from torchrl.envs.libs.vmas import VmasEnv
        >>> env = VmasEnv("balance", num_envs=3, n_agents=5)
        >>> td = env.rand_step()
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([3, 5, 2]))},
                    batch_size=torch.Size([3, 5])),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                info: TensorDict(
                                    fields={
                                        ground_rew: Tensor(shape=torch.Size([3, 5, 1])),
                                        pos_rew: Tensor(shape=torch.Size([3, 5, 1]))},
                                    batch_size=torch.Size([3, 5])),
                                observation: Tensor(shape=torch.Size([3, 5, 16])),
                                reward: Tensor(shape=torch.Size([3, 5, 1]))},
                            batch_size=torch.Size([3, 5])),
                        done: Tensor(shape=torch.Size([3, 1]))},
                    batch_size=torch.Size([3]))},
            batch_size=torch.Size([3]))

We can observe that *keys that are shared by all agents*, such as **done** are present in the root tensordict with
batch size `(num_envs,)`, which represents the number of environments simulated.

On the other hand, *keys that are different between agents*, such as **action**, **reward**, **observation**,
and **info** are present in the nested "agents" tensordict with batch size `(num_envs, n_agents)`,
which represents the additional agent dimension.

Multi-agent tensor specs will follow the same style as in tensordicts.
Specs relating to values that vary between agents will need to be nested in the "agents" entry.

Here is an example of how specs can be created in a multi-agent environment where
only the done flag is shared across agents (as in VMAS):

.. code-block::
   :caption: Example of multi-agent spec creation

        >>> action_specs = []
        >>> observation_specs = []
        >>> reward_specs = []
        >>> info_specs = []
        >>> for i in range(env.n_agents):
        ...    action_specs.append(agent_i_action_spec)
        ...    reward_specs.append(agent_i_reward_spec)
        ...    observation_specs.append(agent_i_observation_spec)
        >>> env.action_spec = CompositeSpec(
        ...    {
        ...        "agents": CompositeSpec(
        ...            {"action": torch.stack(action_specs)}, shape=(env.n_agents,)
        ...        )
        ...    }
        ...)
        >>> env.reward_spec = CompositeSpec(
        ...    {
        ...        "agents": CompositeSpec(
        ...            {"reward": torch.stack(reward_specs)}, shape=(env.n_agents,)
        ...        )
        ...    }
        ...)
        >>> env.observation_spec = CompositeSpec(
        ...    {
        ...        "agents": CompositeSpec(
        ...            {"observation": torch.stack(observation_specs)}, shape=(env.n_agents,)
        ...        )
        ...    }
        ...)
        >>> env.done_spec = DiscreteTensorSpec(
        ...    n=2,
        ...    shape=torch.Size((1,)),
        ...    dtype=torch.bool,
        ... )

As you can see, it is very simple! Per-agent keys will have the nested composite spec and shared keys will follow
single agent standards.

.. note::
  Since reward, done and action keys may have the additional "agent" prefix (e.g., `("agents","action")`),
  the default keys used in the arguments of other TorchRL components (e.g. "action") will not match exactly.
  Therefore, TorchRL provides the `env.action_key`, `env.reward_key`, and `env.done_key` attributes,
  which will automatically point to the right key to use. Make sure you pass these attributes to the various
  components in TorchRL to inform them of the right key (e.g., the `loss.set_keys()` function).

.. note::
  TorchRL abstracts these nested specs away for ease of use.
  This means that accessing `env.reward_spec` will always return the leaf
  spec if the accessed spec is Composite. Therefore, if in the example above
  we run `env.reward_spec` after env creation, we would get the same output as `torch.stack(reward_specs)}`.
  To get the full composite spec with the "agents" key, you can run
  `env.output_spec["full_reward_spec"]`. The same is valid for action and done specs.
  Note that `env.reward_spec == env.output_spec["full_reward_spec"][env.reward_key]`.


.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    MarlGroupMapType
    check_marl_grouping



Transforms
----------

.. _transforms:

.. currentmodule:: torchrl.envs.transforms

In most cases, the raw output of an environment must be treated before being passed to another object (such as a
policy or a value operator). To do this, TorchRL provides a set of transforms that aim at reproducing the transform
logic of `torch.distributions.Transform` and `torchvision.transforms`.
Our environment `tutorial <https://pytorch.org/rl/tutorials/pendulum.html>`_
provides more information on how to design a custom transform.

Transformed environments are build using the :class:`TransformedEnv` primitive.
Composed transforms are built using the :class:`Compose` class:

.. code-block::
   :caption: Transformed environment

        >>> base_env = GymEnv("Pendulum-v1", from_pixels=True, device="cuda:0")
        >>> transform = Compose(ToTensorImage(in_keys=["pixels"]), Resize(64, 64, in_keys=["pixels"]))
        >>> env = TransformedEnv(base_env, transform)

Transforms are usually subclasses of :class:`~torchrl.envs.transforms.Transform`, although any
``Callable[[TensorDictBase], TensorDictBase]``.

By default, the transformed environment will inherit the device of the
:obj:`base_env` that is passed to it. The transforms will then be executed on that device.
It is now apparent that this can bring a significant speedup depending on the kind of
operations that is to be computed.

A great advantage of environment wrappers is that one can consult the environment up to that wrapper.
The same can be achieved with TorchRL transformed environments: the ``parent`` attribute will
return a new :class:`TransformedEnv` with all the transforms up to the transform of interest.
Re-using the example above:

.. code-block::
   :caption: Transform parent

        >>> resize_parent = env.transform[-1].parent  # returns the same as TransformedEnv(base_env, transform[:-1])


Transformed environment can be used with vectorized environments.
Since each transform uses a ``"in_keys"``/``"out_keys"`` set of keyword argument, it is
also easy to root the transform graph to each component of the observation data (e.g.
pixels or states etc).

Transforms also have an ``inv`` method that is called before
the action is applied in reverse order over the composed transform chain:
this allows to apply transforms to data in the environment before the action is taken
in the environment. The keys to be included in this inverse transform are passed through the
``"in_keys_inv"`` keyword argument:

.. code-block::
   :caption: Inverse transform

        >>> env.append_transform(DoubleToFloat(in_keys_inv=["action"]))  # will map the action from float32 to float64 before calling the base_env.step

Cloning transforms
~~~~~~~~~~~~~~~~~~

Because transforms appended to an environment are "registered" to this environment
through the ``transform.parent`` property, when manipulating transforms we should keep
in mind that the parent may come and go following what is being done with the transform.
Here are some examples: if we get a single transform from a :class:`Compose` object,
this transform will keep its parent:

    >>> third_transform = env.transform[2]
    >>> assert third_transform.parent is not None

This means that using this transform for another environment is prohibited, as
the other environment would replace the parent and this may lead to unexpected
behviours. Fortunately, the :class:`Transform` class comes with a :func:`clone`
method that will erase the parent while keeping the identity of all the
registered buffers:

    >>> TransformedEnv(base_env, third_transform)  # raises an Exception as third_transform already has a parent
    >>> TransformedEnv(base_env, third_transform.clone())  # works

On a single process or if the buffers are placed in shared memory, this will
result in all the clone transforms to keep the same behaviour even if the
buffers are changed in place (which is what will happen with the :class:`CatFrames`
transform, for instance). In distributed settings, this may not hold and one
should be careful about the expected behaviour of the cloned transforms in this
context.
Finally, notice that indexing multiple transforms from a :class:`Compose` transform
may also result in loss of parenthood for these transforms: the reason is that
indexing a :class:`Compose` transform results in another :class:`Compose` transform
that does not have a parent environment. Hence, we have to clone the sub-transforms
to be able to create this other composition:

    >>> env = TransformedEnv(base_env, Compose(transform1, transform2, transform3))
    >>> last_two = env.transform[-2:]
    >>> assert isinstance(last_two, Compose)
    >>> assert last_two.parent is None
    >>> assert last_two[0] is not transform2
    >>> assert isinstance(last_two[0], type(transform2))  # and the buffers will match
    >>> assert last_two[1] is not transform3
    >>> assert isinstance(last_two[1], type(transform3))  # and the buffers will match

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Transform
    TransformedEnv
    ActionMask
    BinarizeReward
    BurnInTransform
    CatFrames
    CatTensors
    CenterCrop
    ClipTransform
    Compose
    DeviceCastTransform
    DiscreteActionProjection
    DoubleToFloat
    DTypeCastTransform
    EndOfLifeTransform
    ExcludeTransform
    FiniteTensorDictCheck
    FlattenObservation
    FrameSkipTransform
    GrayScale
    gSDENoise
    InitTracker
    KLRewardTransform
    NoopResetEnv
    ObservationNorm
    ObservationTransform
    PermuteTransform
    PinMemoryTransform
    R3MTransform
    RandomCropTensorDict
    RenameTransform
    Resize
    RewardClipping
    RewardScaling
    RewardSum
    Reward2GoTransform
    RemoveEmptySpecs
    SelectTransform
    SignTransform
    SqueezeTransform
    StepCounter
    TargetReturn
    TensorDictPrimer
    TimeMaxPool
    ToTensorImage
    UnsqueezeTransform
    VecGymEnvTransform
    VecNorm
    VC1Transform
    VIPRewardTransform
    VIPTransform

Environments with masked actions
--------------------------------

In some environments with discrete actions, the actions available to the agent might change throughout execution.
In such cases the environments will output an action mask (under the ``"action_mask"`` key by default).
This mask needs to be used to filter out unavailable actions for that step.

If you are using a custom policy you can pass this mask to your probability distribution like so:

.. code-block::
   :caption: Categorical policy with action mask

        >>> from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, TensorDictSequential
        >>> import torch.nn as nn
        >>> from torchrl.modules import MaskedCategorical
        >>> module = TensorDictModule(
        >>>     nn.Linear(in_feats, out_feats),
        >>>     in_keys=["observation"],
        >>>     out_keys=["logits"],
        >>> )
        >>> dist = ProbabilisticTensorDictModule(
        >>>     in_keys={"logits": "logits", "mask": "action_mask"},
        >>>     out_keys=["action"],
        >>>     distribution_class=MaskedCategorical,
        >>> )
        >>> actor = TensorDictSequential(module, dist)

If you want to use a default policy, you will need to wrap your environment in the :class:`~torchrl.envs.transforms.ActionMask`
transform. This transform can take care of updating the action mask in the action spec in order for the default policy
to always know what the latest available actions are. You can do this like so:

.. code-block::
   :caption: How to use the action mask transform

        >>> from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, TensorDictSequential
        >>> import torch.nn as nn
        >>> from torchrl.envs.transforms import TransformedEnv, ActionMask
        >>> env = TransformedEnv(
        >>>     your_base_env
        >>>     ActionMask(action_key="action", mask_key="action_mask"),
        >>> )

.. note::
  In case you are using a parallel environment it is important to add the transform to the parallel enviornment itself
  and not to its sub-environments.



Recorders
---------

.. currentmodule:: torchrl.record

Recorders are transforms that register data as they come in, for logging purposes.

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    TensorDictRecorder
    VideoRecorder


Helpers
-------
.. currentmodule:: torchrl.envs.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    step_mdp
    get_available_libraries
    set_exploration_mode #deprecated
    set_exploration_type
    exploration_mode #deprecated
    exploration_type
    check_env_specs
    make_composite_from_td
    terminated_or_truncated

Domain-specific
---------------
.. currentmodule:: torchrl.envs

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    ModelBasedEnvBase
    model_based.dreamer.DreamerEnv


Libraries
---------

.. currentmodule:: torchrl.envs

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
    <module 'gymnasium' from '/path/to/venv/python3.9/site-packages/gymnasium/__init__.py'>
    >>> with set_gym_backend(gym):
    ...     print(gym_backend())
    ...     env2 = GymEnv("Pendulum-v1")
    <module 'gym' from '/path/to/venv/python3.9/site-packages/gym/__init__.py'>
    >>> print(env1._env.env.env)
    <gymnasium.envs.classic_control.pendulum.PendulumEnv at 0x15147e190>
    >>> print(env2._env.env.env)
    <gym.envs.classic_control.pendulum.PendulumEnv at 0x1629916a0>

We can see that the two libraries modify the value returned by :func:`~.gym.gym_backend()`
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
        <module 'gym.wrappers' from '/path/to/venv/python3.9/site-packages/gym/wrappers/__init__.py'>
        >>> with set_gym_backend("gymnasium"):
        ...     wrappers = gym_backend('wrappers')
        ...     print(wrappers)
        <module 'gymnasium.wrappers' from '/path/to/venv/python3.9/site-packages/gymnasium/wrappers/__init__.py'>

Another tool that comes in handy with gym and other external dependencies is
the :class:`torchrl._utils.implement_for` class. Decorating a function
with ``@implement_for`` will tell torchrl that, depending on the version
indicated, a specific behaviour is to be expected. This allows us to easily
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
    JumanjiEnv
    JumanjiWrapper
    MOGymEnv
    MOGymWrapper
    MultiThreadedEnv
    MultiThreadedEnvWrapper
    OpenMLEnv
    PettingZooEnv
    PettingZooWrapper
    RoboHiveEnv
    SMACv2Env
    SMACv2Wrapper
    VmasEnv
    VmasWrapper
    gym_backend
    set_gym_backend
