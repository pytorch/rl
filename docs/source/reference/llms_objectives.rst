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

Distillation
------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DistillationLoss
    DistillationLossOutput
    k3_kl_token_estimate
