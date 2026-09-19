# Trainer Basics

Core trainer classes and builder utilities.

## Trainer and hooks

| [`Trainer`](generated/torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)(*args, **kwargs) | A generic Trainer class. |
| --- | --- |
| [`TrainerHookBase`](generated/torchrl.trainers.TrainerHookBase.html#torchrl.trainers.TrainerHookBase)() | An abstract hooking class for torchrl Trainer class. |
| [`MixedPrecisionOptimizationStepper`](generated/torchrl.trainers.MixedPrecisionOptimizationStepper.html#torchrl.trainers.MixedPrecisionOptimizationStepper)(optimizer, *) | Optimization step with mixed precision and gradient accumulation. |

## Algorithm-specific trainers

### On-policy telemetry

On-policy trainers expose `telemetry="standard"` by default. Standard mode
adds diagnostics under the `training/` logger namespace for collected and
batch frames, completed episodes, terminal rates, reward and complete-episode
summaries, optimizer learning rate and gradient norm, collection and optimizer
throughput, and cheap collector or replay-buffer statistics when those values
are available. Optional metrics are omitted when the collected batch does not
contain enough information to compute them; for example, complete-episode
returns require trajectory identifiers and reset markers.

Standard mode uses `training/rewards/{min,mean,std,max}` for transition reward
summaries, without emitting legacy reward or terminal aliases. Synchronous
training summarizes the collected batch. Fully asynchronous training summarizes
valid transitions in replay samples instead, since no collected batch reaches
the learner. Episode and terminal metrics are omitted in that mode: replay slices
may be incomplete or carry artificial boundaries used for advantage estimation.

Set `telemetry="minimal"` to retain the legacy metric set without querying
collector or replay statistics or computing the additional reductions. Legacy
metric names such as `r_training` and `done_percentage` are emitted only in
minimal mode.

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
| [`GRPOTrainer`](generated/torchrl.trainers.algorithms.GRPOTrainer.html#torchrl.trainers.algorithms.GRPOTrainer)(*args, **kwargs) | A trainer for LLM alignment using GRPO (or compatible) objectives. |

## PPO from an environment

`PPOTrainer.from_env()` builds the standard collector, clipped PPO loss,
Adam optimizer, GAE and minibatches. Supply the networks and training budget:

```
trainer = PPOTrainer.from_env(
 env, actor=actor, critic=critic,
 total_frames=1_000_000,
 frames_per_batch=4096,
 minibatch_size=256,
)
trainer.train()
```

For a recurrent policy, keep consecutive time windows and, when appropriate,
normalize advantages separately within each task:

```
trainer = PPOTrainer.from_env(
 env, actor=actor, critic=critic,
 total_frames=1_000_000,
 frames_per_batch=4096,
 minibatch_size=256,
 sub_traj_len=64,
 gae_kwargs={"group_key": "task_id", "average_gae": True},
)
```

The collector installs missing policy primers and initialization tracking.
The environment's unique action and reward keys are inferred, including nested
keys; `value_key` selects the critic output. Episode boundaries default to
siblings of the reward, falling back to root-level keys. Pass explicit trainer
key arguments when a task needs different boundaries. Multi-agent rewards and
boundaries must have compatible shapes, as required by GAE.

`sub_traj_len` counts consecutive steps per environment, while
`frames_per_batch` and `minibatch_size` count transitions across all
environments. Feedforward PPO shuffles individual transitions; recurrent PPO
samples contiguous windows. Training closes the collector's environment: create
a fresh environment for evaluation.

Existing logging and checkpoint options are forwarded to the trainer. Call
`trainer.load_from_file(path)` before `train()` to resume a saved run.
Use the ordinary constructor when supplying custom training components.

Hydra can target the same factory without duplicating its defaults:

```
_target_: torchrl.trainers.algorithms.PPOTrainer.from_env
total_frames: 1000000
frames_per_batch: 4096
minibatch_size: 256
sub_traj_len: 64
gae_kwargs:
 group_key: task_id
 average_gae: true
```

Instantiate it with `instantiate(cfg, env=env, actor=actor, critic=critic)`.
The existing `PPOTrainerConfig`
continues to support explicit component configuration.

*classmethod*PPOTrainer.from_env(*env: [EnvBase](generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*, ***, *actor: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *critic: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *total_frames: int*, *frames_per_batch: int = 1024*, *minibatch_size: int = 256*, *sub_traj_len: int | None = None*, *learning_rate: float = 0.0003*, *value_key: NestedKey = 'state_value'*, *loss_kwargs: Mapping[str, Any] | None = None*, *gae_kwargs: Mapping[str, Any] | None = None*, *collector_kwargs: Mapping[str, Any] | None = None*, ***trainer_kwargs: Any*) → [PPOTrainer](generated/torchrl.trainers.algorithms.PPOTrainer.html#torchrl.trainers.algorithms.PPOTrainer)[[source]](../_modules/torchrl/trainers/algorithms/ppo.html#PPOTrainer.from_env)

Build a PPO trainer from an environment, actor and critic.

Constructs a [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector),
[`ClipPPOLoss`](generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss), Adam optimizer, GAE and
minibatch sampling. Use the ordinary constructor to supply custom
collectors, losses, optimizers or replay buffers.

Parameters:

- **env** ([*EnvBase*](generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - Environment owned by the resulting collector. Training
closes it. The collector installs missing policy primers and
initialization tracking by default.
- **actor** (*TensorDictModuleBase*) - Probabilistic actor returning actions
and their log probabilities.
- **critic** (*TensorDictModuleBase*) - Value network writing `value_key`.
- **total_frames** (*int*) - Total environment transitions to collect. For
closed-loop action deployment these count high-level decisions.
- **frames_per_batch** (*int**,**optional*) - Transitions collected per update,
across all environments. Defaults to 1024.
- **minibatch_size** (*int**,**optional*) - Transitions per optimization step.
Clamped to the collected batch size. Defaults to 256.
- **sub_traj_len** (*int**,**optional*) - Consecutive time steps per recurrent
training window. When set, uses [`BatchSubSampler`](generated/torchrl.trainers.BatchSubSampler.html#torchrl.trainers.BatchSubSampler)
and recurrent-mode GAE instead of flattening time into replay.
The minibatch size must be a multiple of this length.
Defaults to `None` (feedforward PPO).
- **learning_rate** (*float**,**optional*) - Adam learning rate. Defaults to 3e-4.
- **value_key** (*NestedKey**,**optional*) - Critic output key, also configured on
the loss and GAE. Defaults to `"state_value"`.
- **loss_kwargs** (*Mapping**,**optional*) - Extra `ClipPPOLoss` arguments.
Advantage normalization defaults to `True`, or `False` when
GAE already normalizes advantages (e.g. within each task).
- **gae_kwargs** (*Mapping**,**optional*) - Extra GAE arguments. For per-task
normalization, pass `{"group_key": "task_id", "average_gae": True}`.
Recurrent windows default to `shifted=False, deactivate_vmap=True`
because their value networks may depend on recurrent state that
cannot be reconstructed by shifting observations alone.
- **collector_kwargs** (*Mapping**,**optional*) - Extra `Collector` arguments,
such as `policy_device` and `storing_device`.
- ****trainer_kwargs** - Additional [`OnPolicyTrainer`](generated/torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer) options, such
as `num_epochs`, `gamma`, `lmbda`, logging and checkpointing.
Action/reward keys default to the environment's unique keys.
Done/terminated keys use the reward's namespace when available,
otherwise the root namespace; explicit key overrides take precedence.
`frame_skip` defaults to 1 and `clip_norm` to 1.0.

Returns:

Configured trainer; call `train()` to start learning.

Return type:

[PPOTrainer](generated/torchrl.trainers.algorithms.PPOTrainer.html#torchrl.trainers.algorithms.PPOTrainer)

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule, NormalParamExtractor
>>> from torchrl.modules import ProbabilisticActor, TanhNormal
>>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
>>> env = ContinuousActionVecMockEnv()
>>> obs_dim = env.observation_spec["observation"].shape[-1]
>>> action_dim = env.action_spec.shape[-1]
>>> actor = ProbabilisticActor(
... TensorDictModule(
... torch.nn.Sequential(torch.nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor()),
... in_keys=["observation"], out_keys=["loc", "scale"],
... ),
... in_keys=["loc", "scale"], distribution_class=TanhNormal,
... return_log_prob=True,
... )
>>> critic = TensorDictModule(
... torch.nn.Linear(obs_dim, 1), in_keys=["observation"], out_keys=["state_value"],
... )
>>> trainer = PPOTrainer.from_env(
... env, actor=actor, critic=critic, total_frames=32,
... frames_per_batch=16, minibatch_size=8, progress_bar=False,
... )
>>> trainer.train()
```

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