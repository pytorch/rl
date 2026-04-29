.. currentmodule:: torchrl.modules

Robot Learning
==============

Policy architectures for robot manipulation and imitation learning.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ACTModel

World Models and Model-Based RL
===============================

Modules for model-based reinforcement learning, including world models and dynamics models.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    WorldModelWrapper
    DreamerActor
    ObsEncoder
    ObsDecoder
    RSSMPosterior
    RSSMPrior
    RSSMRollout

PILCO
-----

Components for moment-matching model-based policy search (PILCO).

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    GPWorldModel
    RBFController
