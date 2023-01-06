.. currentmodule:: torchrl.objectives

torchrl.objectives package
==========================

DQN
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DQNLoss
    DistributionalDQNLoss

DDPG
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DDPGLoss

SAC
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    SACLoss

REDQ
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    REDQLoss

PPO
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    PPOLoss
    ClipPPOLoss
    KLPENPPOLoss

A2C
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    A2CLoss

Reinforce
---------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ReinforceLoss

Dreamer
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DreamerActorLoss
    DreamerModelLoss
    DreamerValueLoss


Returns
-------
.. currentmodule:: torchrl.objectives.value

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    GAE
    TDLambdaEstimate
    TDEstimate
    functional.generalized_advantage_estimate
    functional.vec_generalized_advantage_estimate
    functional.vec_td_lambda_return_estimate
    functional.vec_td_lambda_advantage_estimate
    functional.td_lambda_return_estimate
    functional.td_lambda_advantage_estimate
    functional.td_advantage_estimate


Utils
-----
.. currentmodule:: torchrl.objectives

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    distance_loss
    hold_out_net
    hold_out_params
    next_state_value
    SoftUpdate
    HardUpdate
