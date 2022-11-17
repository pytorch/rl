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
  The policy should be coded using a :obj:`TensorDictModule` (or any other
  :obj:`TensorDict`-compatible module).


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    EnvBase
    GymLikeEnv
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
    RewardClipping
    Resize
    CenterCrop
    GrayScale
    Compose
    ToTensorImage
    ObservationNorm
    FlattenObservation
    UnsqueezeTransform
    RewardScaling
    ObservationTransform
    CatFrames
    FiniteTensorDictCheck
    DoubleToFloat
    CatTensors
    NoopResetEnv
    BinarizeReward
    PinMemoryTransform
    VecNorm
    gSDENoise
    TensorDictPrimer
    R3MTransform
    VIPTransform

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
