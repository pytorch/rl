.. currentmodule:: torchrl.modules

Exploration Strategies
======================

Exploration modules add noise to actions to enable exploration during training.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    AdditiveGaussianModule
    ConsistentDropoutModule
    EGreedyModule
    OrnsteinUhlenbeckProcessModule

Helpers
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    set_exploration_modules_spec_from_env
