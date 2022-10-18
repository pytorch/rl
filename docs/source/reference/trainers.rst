.. currentmodule:: torchrl.trainers

torchrl.trainers package
======================

Trainer and hooks
-----------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Trainer
    BatchSubSampler
    CountFramesLog
    LogReward
    Recorder
    ReplayBuffer
    RewardNormalizer
    SelectKeys
    UpdateWeights
    ClearCudaCache


Builders
--------

.. currentmodule:: torchrl.trainers.helpers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    make_trainer
    sync_sync_collector
    sync_async_collector
    make_collector_offpolicy
    make_collector_onpolicy
    transformed_env_constructor
    parallel_env_constructor
    make_sac_loss
    make_dqn_loss
    make_ddpg_loss
    make_target_updater
    make_ppo_loss
    make_redq_loss
    make_dqn_actor
    make_ddpg_actor
    make_ppo_model
    make_sac_model
    make_redq_model
    make_replay_buffer

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    correct_for_frame_skip
    get_stats_random_rollout
