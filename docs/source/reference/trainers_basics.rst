.. currentmodule:: torchrl.trainers

Trainer Basics
==============

Core trainer classes and builder utilities.

Trainer and hooks
-----------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Trainer
    TrainerHookBase

Algorithm-specific trainers
---------------------------

.. currentmodule:: torchrl.trainers.algorithms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    PPOTrainer
    SACTrainer

Builders
--------

.. currentmodule:: torchrl.trainers.helpers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    make_collector_offpolicy
    make_collector_onpolicy
    make_dqn_loss
    make_replay_buffer
    make_target_updater
    make_trainer
    parallel_env_constructor
    sync_async_collector
    sync_sync_collector
    transformed_env_constructor

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    correct_for_frame_skip
    get_stats_random_rollout
