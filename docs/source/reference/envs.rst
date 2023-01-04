.. currentmodule:: torchrl.envs

torchrl.envs package
====================

TorchRL offers an API to handle environments of different backends, such as gym,
dm-control, dm-lab, model-based environments as well as custom environments.
The goal is to be able to swap environments in an experiment with little or no effort,
even if these environments are simulated using different libraries.
TorchRL offers some out-of-the-box environment wrappers under :obj:`torchrl.envs.libs`,
which we hope can be easily imitated for other libraries.
The parent class :obj:`EnvBase` is a :obj:`torch.nn.Module` subclass that implements
some typical environment methods using :obj:`TensorDict` as a data organiser. This allows this
class to be generic and to handle an arbitrary number of input and outputs, as well as
nested or batched data structures.

Each env will have the following attributes:

- :obj:`env.batch_size`: a :obj:`torch.Size` representing the number of envs batched together.
- :obj:`env.device`: the device where the input and output tensordict are expected to live.
  The environment device does not mean that the actual step operations will be computed on device
  (this is the responsibility of the backend, with which TorchRL can do little). The device of
  an environment just represents the device where the data is to be expected when input to the
  environment or retrieved from it. TorchRL takes care of mapping the data to the desired device.
  This is especially useful for transforms (see below). For parametric environments (e.g.
  model-based environments), the device does represent the hardware that will be used to
  compute the operations.
- :obj:`env.observation_spec`: a :obj:`CompositeSpec` object containing all the observation key-spec pairs.
- :obj:`env.input_spec`: a :obj:`CompositeSpec` object containing all the input keys (:obj:`"action"` and others).
- :obj:`env.action_spec`: a :obj:`TensorSpec` object representing the action spec.
- :obj:`env.reward_spec`: a :obj:`TensorSpec` object representing the reward spec.

Importantly, the environment spec shapes should *not* contain the batch size, e.g.
an environment with :obj:`env.batch_size == torch.Size([4])` should not have
an :obj:`env.action_spec` with shape :obj:`torch.Size([4, action_size])` but simply
:obj:`torch.Size([action_size])`.

With these, the following methods are implemented:

- :obj:`env.reset(tensordict)`: a reset method that may (but not necessarily requires to) take
  a :obj:`TensorDict` input. It return the first tensordict of a rollout, usually
  containing a :obj:`"done"` state and a set of observations.
- :obj:`env.step(tensordict)`: a step method that takes a :obj:`TensorDict` input
  containing an input action as well as other inputs (for model-based or stateless
  environments, for instance).
- :obj:`env.set_seed(integer)`: a seeding method that will return the next seed
  to be used in a multi-env setting. This next seed is deterministically computed
  from the preceding one, such that one can seed multiple environments with a different
  seed without risking to overlap seeds in consecutive experiments, while still
  having reproducible results.
- :obj:`env.rollout(max_steps, policy)`: executes a rollout in the environment for
  a maximum number of steps :obj:`max_steps` and using a policy :obj:`policy`.
  The policy should be coded using a :obj:`SafeModule` (or any other
  :obj:`TensorDict`-compatible module).


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    EnvBase
    GymLikeEnv

Vectorized envs
---------------

Vectorized (or better: parallel) environments is a common feature in Reinforcement Learning
where executing the environment step can be cpu-intensive.
Some libraries such as `gym3 <https://github.com/openai/gym3>`_ or `EnvPool <https://github.com/sail-sg/envpool>`_
offer interfaces to execute batches of environments simultaneously.
While they often offer a very competitive computational advantage, they do not
necessarily scale to the wide variety of environment libraries supported by TorchRL.
Therefore, TorchRL offers its own, generic :obj:`ParallelEnv` class to run multiple
environments in parallel.
As this class inherits from :obj:`EnvBase`, it enjoys the exact same API as other environment.
Of course, a :obj:`ParallelEnv` will have a batch size that corresponds to its environment count:

.. code-block::
   :caption: Parallel environment

        >>> def make_env():
        ...     return GymEnv("Pendulum-v1", from_pixels=True, g=9.81, device="cuda:0")
        >>> env = ParallelEnv(4, make_env)
        >>> print(env.batch_size)
        torch.Size([4])

:obj:`ParallelEnv` allows to retrieve the attributes from its contained environments:
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


*A note on performance*: launching a :obj:`ParallelEnv` can take quite some time
as it requires to launch as many python instances as there are processes. Due to
the time that it takes to run :obj:`import torch` (and other imports), starting the
parallel env can be a bottleneck. This is why, for instance, TorchRL tests are so slow.
Once the environment is launched, a great speedup should be observed.

Another thing to take in consideration is that :obj:`ParallelEnv`s (as well as data collectors)
will create data buffers based on the environment specs to pass data from one process
to another. This means that a misspecified spec (input, observation or reward) will
cause a breakage at runtime as the data can't be written on the preallocated buffer.
In general, an environment should be tested using the :obj:`check_env_specs`
test function before being used in a :obj:`ParallelEnv`. This function will raise
an assertion error whenever the preallocated buffer and the collected data mismatch.

We also offer the :obj:`SerialEnv` class that enjoys the exact same API but is executed
serially. This is mostly useful for testing purposes, when one wants to assess the
behaviour of a :obj:`ParallelEnv` without launching the subprocesses.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    SerialEnv
    ParallelEnv


Transforms
----------
.. currentmodule:: torchrl.envs.transforms

In most cases, the raw output of an environment must be treated before being passed to another object (such as a
policy or a value operator). To do this, TorchRL provides a set of transforms that aim at reproducing the transform
logic of `torch.distributions.Transform` and `torchvision.transforms`.

Transformed environments are build using the :doc:`TransformedEnv` primitive.
Composed transforms are built using the :doc:`Compose` class:

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
return a new :obj:`TransformedEnv` with all the transforms up to the transform of interest.
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
    DoubleToFloat
    FiniteTensorDictCheck
    FlattenObservation
    FrameSkipTransform
    GrayScale
    gSDENoise
    NoopResetEnv
    ObservationNorm
    ObservationTransform
    PinMemoryTransform
    Resize
    RewardClipping
    RewardScaling
    RewardSum
    SqueezeTransform
    StepCounter
    TensorDictPrimer
    ToTensorImage
    UnsqueezeTransform
    VecNorm
    R3MTransform
    VIPTransform
    VIPRewardTransform

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

    gym.GymEnv
    gym.GymWrapper
    dm_control.DMControlEnv
    dm_control.DMControlWrapper
    jumanji.JumanjiEnv
    jumanji.JumanjiWrapper
    habitat.HabitatEnv
