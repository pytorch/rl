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

TorchRL distributions returned by probabilistic modules expose joint
sample-and-score methods. This lets consumers score the exact draw without
depending on a particular distribution class::

    action, log_prob = dist.rsample_and_log_prob()

The :func:`ensure_rsample_and_log_prob` adapter preserves object identity for
distributions that already implement this method and transparently wraps other
PyTorch or third-party distributions. Adapted distributions preserve
``isinstance`` behavior and can be passed directly to
:func:`torch.distributions.kl_divergence`. The free functions remain available
for raw distributions.

.. currentmodule:: torchrl.modules.distributions.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    sample_and_log_prob
    rsample_and_log_prob
    ensure_rsample_and_log_prob
    composite_entropy
