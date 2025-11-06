.. currentmodule:: torchrl.objectives

Common Components
=================

Base classes and common utilities for all loss modules.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    LossModule
    add_random_module

Value Estimators
----------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ValueEstimatorBase
    ValueEstimators
    TD0Estimator
    TD1Estimator
    TDLambdaEstimator
    GAE
