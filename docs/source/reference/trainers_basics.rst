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
    MixedPrecisionOptimizationStepper

Algorithm-specific trainers
---------------------------

.. currentmodule:: torchrl.trainers.algorithms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    OnPolicyTrainer
    A2CTrainer
    PPOTrainer
    ReinforceTrainer
    SACTrainer
    OfflineToOnlineTrainer
    DQNTrainer
    DDPGTrainer
    IQLTrainer
    CQLTrainer
    TD3Trainer
    GRPOTrainer

PPO from an environment
-----------------------

:meth:`PPOTrainer.from_env` builds the standard collector, clipped PPO loss,
Adam optimizer, GAE and minibatches. Supply the networks and training budget::

    trainer = PPOTrainer.from_env(
        env, actor=actor, critic=critic,
        total_frames=1_000_000,
        frames_per_batch=4096,
        minibatch_size=256,
    )
    trainer.train()

For a recurrent policy, keep consecutive time windows and, when appropriate,
normalize advantages separately within each task::

    trainer = PPOTrainer.from_env(
        env, actor=actor, critic=critic,
        total_frames=1_000_000,
        frames_per_batch=4096,
        minibatch_size=256,
        sub_traj_len=64,
        gae_kwargs={"group_key": "task_id", "average_gae": True},
    )

The collector installs missing policy primers and initialization tracking.
The environment's unique action and reward keys are inferred, including nested
keys; ``value_key`` selects the critic output. Episode boundaries default to
siblings of the reward, falling back to root-level keys. Pass explicit trainer
key arguments when a task needs different boundaries. Multi-agent rewards and
boundaries must have compatible shapes, as required by GAE.

``sub_traj_len`` counts consecutive steps per environment, while
``frames_per_batch`` and ``minibatch_size`` count transitions across all
environments. Feedforward PPO shuffles individual transitions; recurrent PPO
samples contiguous windows. Training closes the collector's environment: create
a fresh environment for evaluation.

Existing logging and checkpoint options are forwarded to the trainer. Call
``trainer.load_from_file(path)`` before ``train()`` to resume a saved run.
Use the ordinary constructor when supplying custom training components.

Hydra can target the same factory without duplicating its defaults:

.. code-block:: yaml

    _target_: torchrl.trainers.algorithms.PPOTrainer.from_env
    total_frames: 1000000
    frames_per_batch: 4096
    minibatch_size: 256
    sub_traj_len: 64
    gae_kwargs:
      group_key: task_id
      average_gae: true

Instantiate it with ``instantiate(cfg, env=env, actor=actor, critic=critic)``.
The existing :class:`~torchrl.trainers.algorithms.configs.PPOTrainerConfig`
continues to support explicit component configuration.

.. automethod:: PPOTrainer.from_env

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
