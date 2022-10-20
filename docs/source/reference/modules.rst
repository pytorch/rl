.. currentmodule:: torchrl.modules

torchrl.modules package
=======================

TensorDict modules
------------------


.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    TensorDictModule
    ProbabilisticTensorDictModule
    TensorDictSequential
    TensorDictModuleWrapper
    Actor
    ProbabilisticActor
    ValueOperator
    QValueActor
    DistributionalQValueActor
    ActorValueOperator
    ActorCriticOperator
    ActorCriticWrapper

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


Distributions
-------------
.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Delta
    TanhNormal
    TruncatedNormal
    TanhDelta
    OneHotCategorical

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    mappings
    inv_softplus
    biased_softplus
    SqueezeLayer
    Squeeze2dLayer
