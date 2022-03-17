.. currentmodule:: torchrl.agents

torchrl.agents package
======================

Agents
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Agent
    EnvCreator


Builders
--------

.. currentmodule:: torchrl.agents.helpers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    make_agent
    sync_sync_collector
    sync_async_collector
    make_collector_offpolicy
    make_collector_onpolicy
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
    transformed_env_constructor
    parallel_env_constructor

Utils
-----


.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    correct_for_frame_skip
    get_stats_random_rollout

Argument parser
---------------


.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    parser_agent_args
    parser_collector_args_offpolicy
    parser_collector_args_onpolicy
    parser_env_args
    parser_loss_args
    parser_loss_args_ppo
    parser_model_args_continuous
    parser_model_args_discrete
    parser_recorder_args
    parser_replay_args
