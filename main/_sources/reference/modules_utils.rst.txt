.. currentmodule:: torchrl.modules

Utilities and Helpers
=====================

Utility modules and helper functions for building RL networks.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ActorValueOperator
    ActorCriticOperator
    ActorCriticWrapper
    get_primers_from_module
    get_env_transforms_from_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    get_recurrent_matmul_precision
    set_recurrent_matmul_precision

.. autodata:: RecurrentMatmulPrecision

.. autodata:: RecurrentMatmulPrecisionUserMode

.. currentmodule:: torchrl.modules.models.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    SquashDims
