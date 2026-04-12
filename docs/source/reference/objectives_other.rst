.. currentmodule:: torchrl.objectives

Other Loss Modules
==================

Additional loss modules for specialized algorithms.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    GAILLoss
    DTLoss
    OnlineDTLoss
    DreamerActorLoss
    DreamerModelLoss
    DreamerValueLoss
    ExponentialQuadraticCost

DreamerV3
---------

Loss modules for DreamerV3 (`Mastering Diverse Domains in World Models, Hafner et al. 2023 <https://arxiv.org/abs/2301.04104>`_).
Key differences from V1: discrete categorical latent state, KL balancing, symlog transforms, and two-hot value distributions.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DreamerV3ActorLoss
    DreamerV3ModelLoss
    DreamerV3ValueLoss

DreamerV3 Utilities
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    symlog
    symexp
    two_hot_encode
    two_hot_decode
