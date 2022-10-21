.. currentmodule:: torchrl.envs

torchrl.envs package
====================

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
