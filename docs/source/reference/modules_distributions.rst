.. currentmodule:: torchrl.modules

Distribution Classes
====================

Custom distribution classes for RL, extending PyTorch distributions.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Delta
    IndependentNormal
    MaskedCategorical
    NormalParamExtractor
    OneHotCategorical
    ReparamGradientStrategy
    TanhDelta
    TanhNormal
    TruncatedNormal

Sampling utilities
==================

.. currentmodule:: torchrl.modules.distributions.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    sample_and_log_prob
    rsample_and_log_prob
    composite_entropy
    has_analytic_entropy
    has_analytic_kl
