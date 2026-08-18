:orphan:

.. currentmodule:: torchrl.objectives.llm

LLM Objectives
==============

Specialized loss functions for LLM training.

GRPO
----

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    GRPOLoss
    GRPOLossOutput
    MCAdvantage
    MCAdvantageSelector
    RayMCAdvantage

SFT
---

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    SFTLoss
    SFTLossOutput

Reward Model Training
---------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RewardModelLoss
    RewardModelLossOutput

Distillation
------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DistillationLoss
    DistillationLossOutput
    k3_kl_token_estimate
