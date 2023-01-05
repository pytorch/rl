.. currentmodule:: torchrl.modules

torchrl.modules package
=======================

TensorDict modules
------------------
.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Actor
    ActorCriticOperator
    ActorCriticWrapper
    ActorValueOperator
    DistributionalQValueActor
    ProbabilisticActor
    QValueActor
    ValueOperator
    SafeModule
    AdditiveGaussianWrapper
    EGreedyWrapper
    OrnsteinUhlenbeckProcessWrapper
    SafeProbabilisticModule
    SafeProbabilisticSequential
    SafeSequential
    WorldModelWrapper
    common.is_tensordict_compatible
    common.ensure_tensordict_compatible

Hooks
-----
.. currentmodule:: torchrl.modules.tensordict_module.actors

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    QValueHook
    DistributionalQValueHook

Models
------
.. currentmodule:: torchrl.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    MLP
    ConvNet
    DuelingCnnDQNet
    DistributionalDQNnet
    DdpgCnnActor
    DdpgCnnQNet
    DdpgMlpActor
    DdpgMlpQNet
    LSTMNet
    DreamerActor
    ObsEncoder
    ObsDecoder
    RSSMPrior
    RSSMPosterior


Exploration
-----------
.. currentmodule:: torchrl.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    NoisyLinear
    NoisyLazyLinear
    reset_noise


Planners
--------
.. currentmodule:: torchrl.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    CEMPlanner
    MPCPlannerBase
    MPPIPlanner


Distributions
-------------
.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Delta
    IndependentNormal
    NormalParamWrapper
    TanhNormal
    TruncatedNormal
    TanhDelta
    OneHotCategorical

Utils
-----

.. currentmodule:: torchrl.modules.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    mappings
    inv_softplus
    biased_softplus

.. currentmodule:: torchrl.modules.models.utils

    SqueezeLayer
    Squeeze2dLayer
