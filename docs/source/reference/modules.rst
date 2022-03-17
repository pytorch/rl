.. currentmodule:: torchrl.modules

torchrl.modules package
=======================

TensorDict modules
------------------


.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    TDModule
    ProbabilisticTDModule
    TDSequence
    TDModuleWrapper
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
.. currentmodule:: torchrl.modules.td_module.actors

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
