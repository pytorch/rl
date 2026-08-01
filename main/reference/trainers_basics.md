# Trainer Basics

Core trainer classes and builder utilities.

## Trainer and hooks

| [`Trainer`](generated/torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)(*args, **kwargs) | A generic Trainer class. |
| --- | --- |
| [`TrainerHookBase`](generated/torchrl.trainers.TrainerHookBase.html#torchrl.trainers.TrainerHookBase)() | An abstract hooking class for torchrl Trainer class. |

## Algorithm-specific trainers

| [`OnPolicyTrainer`](generated/torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer)(*args, **kwargs) | Shared implementation for on-policy trainers (PPO, A2C, REINFORCE). |
| --- | --- |
| [`A2CTrainer`](generated/torchrl.trainers.algorithms.A2CTrainer.html#torchrl.trainers.algorithms.A2CTrainer)(*args, **kwargs) | A2C (Advantage Actor-Critic) trainer implementation. |
| [`PPOTrainer`](generated/torchrl.trainers.algorithms.PPOTrainer.html#torchrl.trainers.algorithms.PPOTrainer)(*args, **kwargs) | PPO (Proximal Policy Optimization) trainer implementation. |
| [`ReinforceTrainer`](generated/torchrl.trainers.algorithms.ReinforceTrainer.html#torchrl.trainers.algorithms.ReinforceTrainer)(*args, **kwargs) | REINFORCE (policy gradient with baseline) trainer implementation. |
| [`SACTrainer`](generated/torchrl.trainers.algorithms.SACTrainer.html#torchrl.trainers.algorithms.SACTrainer)(*args, **kwargs) | A trainer class for Soft Actor-Critic (SAC) algorithm. |
| [`OfflineToOnlineTrainer`](generated/torchrl.trainers.algorithms.OfflineToOnlineTrainer.html#torchrl.trainers.algorithms.OfflineToOnlineTrainer)(*args, **kwargs) | A SAC trainer for the offline-pretrain -> online-finetune transition. |
| [`DQNTrainer`](generated/torchrl.trainers.algorithms.DQNTrainer.html#torchrl.trainers.algorithms.DQNTrainer)(*args, **kwargs) | A trainer class for Deep Q-Network (DQN) algorithm. |
| [`DDPGTrainer`](generated/torchrl.trainers.algorithms.DDPGTrainer.html#torchrl.trainers.algorithms.DDPGTrainer)(*args, **kwargs) | A trainer class for Deep Deterministic Policy Gradient (DDPG) algorithm. |
| [`IQLTrainer`](generated/torchrl.trainers.algorithms.IQLTrainer.html#torchrl.trainers.algorithms.IQLTrainer)(*args, **kwargs) | A trainer class for Implicit Q-Learning (IQL) algorithm. |
| [`CQLTrainer`](generated/torchrl.trainers.algorithms.CQLTrainer.html#torchrl.trainers.algorithms.CQLTrainer)(*args, **kwargs) | A trainer class for Conservative Q-Learning (CQL) algorithm. |
| [`TD3Trainer`](generated/torchrl.trainers.algorithms.TD3Trainer.html#torchrl.trainers.algorithms.TD3Trainer)(*args, **kwargs) | A trainer class for Twin Delayed DDPG (TD3) algorithm. |

## Builders

| [`make_collector_offpolicy`](generated/torchrl.trainers.helpers.make_collector_offpolicy.html#torchrl.trainers.helpers.make_collector_offpolicy)(make_env, ...[, ...]) | Returns a data collector for off-policy sota-implementations. |
| --- | --- |
| [`make_collector_onpolicy`](generated/torchrl.trainers.helpers.make_collector_onpolicy.html#torchrl.trainers.helpers.make_collector_onpolicy)(make_env, ...[, ...]) | Makes a collector in on-policy settings. |
| [`make_dqn_loss`](generated/torchrl.trainers.helpers.make_dqn_loss.html#torchrl.trainers.helpers.make_dqn_loss)(model, cfg) | Builds the DQN loss module. |
| [`make_replay_buffer`](generated/torchrl.trainers.helpers.make_replay_buffer.html#torchrl.trainers.helpers.make_replay_buffer)(device, cfg) | Builds a replay buffer using the config built from ReplayArgsConfig. |
| [`make_target_updater`](generated/torchrl.trainers.helpers.make_target_updater.html#torchrl.trainers.helpers.make_target_updater)(cfg, loss_module) | Builds a target network weight update object. |
| [`make_trainer`](generated/torchrl.trainers.helpers.make_trainer.html#torchrl.trainers.helpers.make_trainer)(collector, loss_module[, ...]) | Creates a Trainer instance given its constituents. |
| [`parallel_env_constructor`](generated/torchrl.trainers.helpers.parallel_env_constructor.html#torchrl.trainers.helpers.parallel_env_constructor)(cfg, **kwargs) | Returns a parallel environment from an argparse.Namespace built with the appropriate parser constructor. |
| [`sync_async_collector`](generated/torchrl.trainers.helpers.sync_async_collector.html#torchrl.trainers.helpers.sync_async_collector)(env_fns, env_kwargs[, ...]) | Runs asynchronous collectors, each running synchronous environments. |
| [`sync_sync_collector`](generated/torchrl.trainers.helpers.sync_sync_collector.html#torchrl.trainers.helpers.sync_sync_collector)(env_fns, env_kwargs[, ...]) | Runs synchronous collectors, each running synchronous environments. |
| [`transformed_env_constructor`](generated/torchrl.trainers.helpers.transformed_env_constructor.html#torchrl.trainers.helpers.transformed_env_constructor)(cfg[, ...]) | Returns an environment creator from an argparse.Namespace built with the appropriate parser constructor. |

## Utils

| [`correct_for_frame_skip`](generated/torchrl.trainers.helpers.correct_for_frame_skip.html#torchrl.trainers.helpers.correct_for_frame_skip)(cfg) | Correct the arguments for the input frame_skip, by dividing all the arguments that reflect a count of frames by the frame_skip. |
| --- | --- |
| [`get_stats_random_rollout`](generated/torchrl.trainers.helpers.get_stats_random_rollout.html#torchrl.trainers.helpers.get_stats_random_rollout)(cfg[, ...]) | Gathers stas (loc and scale) from an environment using random rollouts. |