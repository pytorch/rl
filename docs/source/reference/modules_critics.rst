.. currentmodule:: torchrl.modules

Value Networks and Critics
==========================

Value networks estimate the value of states or state-action pairs.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ValueOperator
    ValueNorm
    PopArtValueNorm
    RunningValueNorm
    DuelingCnnDQNet
    DistributionalDQNnet
    ConvNet
    CrossCriticGroupSpec
    CrossGroupCritic
    MLP
    DdpgCnnActor
    DdpgCnnQNet
    DdpgMlpActor
    DdpgMlpQNet
    LSTMModule
    GRUModule
    canonicalize_rnn_subset
    set_recurrent_mode
    OnlineDTActor
    DTActor
    DecisionTransformer

Value transforms
----------------

Value transforms compress large reward and return scales into a numerically
convenient prediction space and provide the inverse mapping needed before
bootstrapping. They can be composed to build custom invertible mappings.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ValueTransform
    IdentityValueTransform
    SymLogValueTransform
    SignedHyperbolicValueTransform
    ComposeValueTransform
    symlog
    symexp
    signed_hyperbolic
    signed_parabolic
