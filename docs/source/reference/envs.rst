.. currentmodule:: torchrl.envs

torchrl.envs package
====================

TorchRL offers an API to handle environments of different backends, such as gym,
dm-control, dm-lab, model-based environments as well as custom environments.
The goal is to be able to swap environments in an experiment with little or no effort,
even if these environments are simulated using different libraries.
TorchRL offers some out-of-the-box environment wrappers under :obj:`torchrl.envs.libs`,
which we hope can be easily imitated for other libraries.
The parent class :class:`torchrl.envs.EnvBase` is a :class:`torch.nn.Module` subclass that implements
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
- :obj:`env.input_spec`: a :class:`torchrl.data.CompositeSpec` object containing
  all the input keys (:obj:`"action"` and others).
- :obj:`env.output_spec`: a :class:`torchrl.data.CompositeSpec` object containing
  all the output keys (:obj:`"observation"`, :obj:`"reward"` and :obj:`"done"`).
- :obj:`env.observation_spec`: a :class:`torchrl.data.CompositeSpec` object
  containing all the observation key-spec pairs.
  This is a pointer to ``env.output_spec["observation"]``.
- :obj:`env.action_spec`: a :class:`torchrl.data.TensorSpec` object
  representing the action spec. This is a pointer to ``env.input_spec["action"]``.
- :obj:`env.reward_spec`: a :class:`torchrl.data.TensorSpec` object representing
  the reward spec. This is a pointer to ``env.output_spec["reward"]``.
- :obj:`env.done_spec`: a :class:`torchrl.data.TensorSpec` object representing
  the done-flag spec. This is a pointer to ``env.output_spec["done"]``.

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
- :meth:`env.set_seed`: a seeding method that will return the next seed
  to be used in a multi-env setting. This next seed is deterministically computed
  from the preceding one, such that one can seed multiple environments with a different
  seed without risking to overlap seeds in consecutive experiments, while still
  having reproducible results.
- :meth:`env.rollout`: executes a rollout in the environment for
  a maximum number of steps (``max_steps=N``) and using a policy (``policy=model``).
  The policy should be coded using a :class:`tensordict.nn.TensorDictModule`
  (or any other :class:`tensordict.TensorDict`-compatible module).

The following figure summarizes how a rollout is executed in torchrl.

.. figure:: /_static/img/rollout.gif

   TorchRL rollouts using TensorDict.

In brief, a TensorDict is created by the :meth:`~.EnvBase.reset` method,
then populated with an action by the policy before being passed to the
:meth:`~.EnvBase.step` method which writes the observations, done flag and
reward under the ``"next"`` entry. The result of this call is stored for
delivery and the ``"next"`` entry is gathered by the :func:`~.utils.step_mdp`
function.

.. note::

  The Gym(nasium) API recently shifted to a splitting of the ``"done"`` state
  into a ``terminated`` (the env is done and results should not be trusted)
  and ``truncated`` (the maximum number of steps is reached) flags.
  In TorchRL, ``"done"`` usually refers to ``"terminated"``. Truncation is
  achieved via the :class:`~.StepCounter` transform class, and the output
  key will be ``"truncated"`` if not chosen to be something else (e.g.
  ``StepCounter(max_steps=100, truncated_key="done")``).
  TorchRL's collectors and rollout methods will be looking for one of these
  keys when assessing if the env should be reset.

.. note::

  The `torchrl.collectors.utils.split_trajectories` function can be used to
  slice adjacent trajectories. It relies on a ``"traj_ids"`` entry in the
  input tensordict, or to the junction of ``"done"`` and ``"truncated"`` key
  if the ``"traj_ids"`` is missing.


.. note::

  In some contexts, it can be useful to mark the first step of a trajectory.
  TorchRL provides such functionality through the :class:`torchrl.envs.InitTracker`
  transform.


Our environment `tutorial <https://pytorch.org/rl/tutorials/pendulum.html>`_
provides more information on how to design a custom environment from scratch.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    EnvBase
    GymLikeEnv
    EnvMetaData
    Specs

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

It is important that your environment specs match the input and output that it sends and receives, as
:class:`ParallelEnv` will create buffers from these specs to communicate with the spawn processes.
Check the :func:`torchrl.envs.utils.check_env_specs` method for a sanity check.

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

It is also possible to reset some but not all of the environments:

.. code-block::
   :caption: Parallel environment reset

        >>> tensordict = TensorDict({"reset_workers": [True, False, True, True]}, [4])
        >>> env.reset(tensordict)
        TensorDict(
            fields={
                done: Tensor(torch.Size([4, 1]), dtype=torch.bool),
                pixels: Tensor(torch.Size([4, 500, 500, 3]), dtype=torch.uint8),
                reset_workers: Tensor(torch.Size([4]), dtype=torch.bool)},
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
    MultiThreadedEnv
    EnvCreator


Transforms
----------
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


By default, the transformed environment will inherit the device of the
:obj:`base_env` that is passed to it. The transforms will then be executed on that device.
It is now apparent that this can bring a significant speedup depending on the kind of
operations that is to be computed.

A great advantage of environment wrappers is that one can consult the environment up to that wrapper.
The same can be achieved with TorchRL transformed environments: the :doc:`parent` attribute will
return a new :class:`TransformedEnv` with all the transforms up to the transform of interest.
Re-using the example above:

.. code-block::
   :caption: Transform parent

        >>> resize_parent = env.transform[-1].parent  # returns the same as TransformedEnv(base_env, transform[:-1])


Transformed environment can be used with vectorized environments.
Since each transform uses a :doc:`"in_keys"`/:doc:`"out_keys"` set of keyword argument, it is
also easy to root the transform graph to each component of the observation data (e.g.
pixels or states etc).

Transforms also have an :doc:`inv` method that is called before
the action is applied in reverse order over the composed transform chain:
this allows to apply transforms to data in the environment before the action is taken
in the environment. The keys to be included in this inverse transform are passed through the
:doc:`"in_keys_inv"` keyword argument:

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
    BinarizeReward
    CatFrames
    CatTensors
    CenterCrop
    Compose
    DiscreteActionProjection
    DoubleToFloat
    ExcludeTransform
    FiniteTensorDictCheck
    FlattenObservation
    FrameSkipTransform
    GrayScale
    gSDENoise
    InitTracker
    NoopResetEnv
    ObservationNorm
    ObservationTransform
    PinMemoryTransform
    R3MTransform
    RandomCropTensorDict
    Resize
    RewardClipping
    RewardScaling
    RewardSum
    SelectTransform
    SqueezeTransform
    StepCounter
    TensorDictPrimer
    TimeMaxPool
    ToTensorImage
    UnsqueezeTransform
    VecNorm
    VIPRewardTransform
    VIPTransform

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
    set_exploration_mode
    exploration_mode
    check_env_specs

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
.. currentmodule:: torchrl.envs.libs

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    brax.BraxEnv
    brax.BraxWrapper
    dm_control.DMControlEnv
    dm_control.DMControlWrapper
    gym.GymEnv
    gym.GymWrapper
    habitat.HabitatEnv
    jumanji.JumanjiEnv
    jumanji.JumanjiWrapper
    vmas.VmasEnv
    vmas.VmasWrapper
