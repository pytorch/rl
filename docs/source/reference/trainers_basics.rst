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

On-policy telemetry
~~~~~~~~~~~~~~~~~~~

On-policy trainers expose ``telemetry="standard"`` by default. Standard mode
adds diagnostics under the ``training/`` logger namespace for collected and
batch frames, completed episodes, terminal rates, reward and complete-episode
summaries, optimizer learning rate and gradient norm, collection and optimizer
throughput, and cheap collector or replay-buffer statistics when those values
are available. Optional metrics are omitted when the collected batch does not
contain enough information to compute them; for example, complete-episode
returns require trajectory identifiers and reset markers.

Standard mode uses ``training/rewards/{min,mean,std,max}`` for transition reward
summaries, without emitting legacy reward or terminal aliases. Synchronous
training summarizes the collected batch. Fully asynchronous training summarizes
valid transitions in replay samples instead, since no collected batch reaches
the learner. Episode and terminal metrics are omitted in that mode: replay slices
may be incomplete or carry artificial boundaries used for advantage estimation.

Set ``telemetry="minimal"`` to retain the legacy metric set without querying
collector or replay statistics or computing the additional reductions. Legacy
metric names such as ``r_training`` and ``done_percentage`` are emitted only in
minimal mode.

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
