.. currentmodule:: torchrl.envs

torchrl.envs package
====================

TorchRL offers an API to handle environments of different backends, such as gym,
dm-control, dm-lab, model-based environments as well as custom environments.
The goal is to be able to swap environments in an experiment with little or no effort,
even if these environments are simulated using different libraries.
The parent class :obj:`EnvBase` is a :obj:`torch.nn.Module` subclass that implements
the typical environment methods using :obj:`TensorDict` instances. This allows this
class to be generic and to handle an arbitrary number of input and outputs, as well as
nested or batched data structures.

Each env will have the following attributes:

- :obj:`env.batch_size`: a :obj:`torch.Size` representing the number of envs batched together.
- :obj:`env.device`: the device where the input and output tensordict are expected to live.
- :obj:`env.observation_spec`: a :obj:`CompositeSpec` object containing all the observation keys.
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


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    EnvBase
    GymLikeEnv
    SerialEnv
    ParallelEnv

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

Transforms
----------
.. currentmodule:: torchrl.envs.transforms

In most cases, the raw output of an environment must be treated before being passed to another object (such as a
policy or a value operator). To do this, TorchRL provides a set of transforms that aim at reproducing the transform
logic of `torch.distributions.Transform` and `torchvision.transforms`.

Transforms can


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
