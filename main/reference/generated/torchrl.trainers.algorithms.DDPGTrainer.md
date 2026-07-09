# DDPGTrainer

*class*torchrl.trainers.algorithms.DDPGTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/ddpg.html#DDPGTrainer)

A trainer class for Deep Deterministic Policy Gradient (DDPG) algorithm.

See also `DDPGTrainerConfig` for the
Hydra configuration counterpart.

This trainer implements the DDPG algorithm, an off-policy actor-critic method
that learns a deterministic policy for continuous action spaces.

The trainer handles:
- Replay buffer management for off-policy learning
- Target network updates (typically SoftUpdate) for stable training
- Policy weight updates to the data collector
- Comprehensive logging of training metrics

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The data collector used to gather environment interactions.
- **total_frames** (*int*) - Total number of frames to collect during training.
- **frame_skip** (*int*) - Number of frames to skip between policy updates.
- **optim_steps_per_batch** (*int*) - Number of optimization steps per collected batch.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)*|**Callable*) - The DDPG loss module.
- **optimizer** (*optim.Optimizer**,**optional*) - The optimizer for training.
- **logger** (*Logger**,**optional*) - Logger for recording training metrics. Defaults to None.
- **clip_grad_norm** (*bool**,**optional*) - Whether to clip gradient norms. Defaults to True.
- **clip_norm** (*float**,**optional*) - Maximum gradient norm for clipping. Defaults to None.
- **progress_bar** (*bool**,**optional*) - Whether to show a progress bar. Defaults to True.
- **seed** (*int**,**optional*) - Random seed for reproducibility. Defaults to None.
- **save_trainer_interval** (*int**,**optional*) - Interval for saving trainer state. Defaults to 10000.
- **log_interval** (*int**,**optional*) - Interval for logging metrics. Defaults to 10000.
- **save_trainer_file** (*str**|**pathlib.Path**,**optional*) - File path for saving trainer state.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - Replay buffer for storing experiences. Defaults to None.
- **enable_logging** (*bool**,**optional*) - Whether to enable metric logging. Defaults to True.
- **log_rewards** (*bool**,**optional*) - Whether to log reward statistics. Defaults to True.
- **log_actions** (*bool**,**optional*) - Whether to log action statistics. Defaults to True.
- **log_observations** (*bool**,**optional*) - Whether to log observation statistics. Defaults to False.
- **target_net_updater** (*TargetNetUpdater**,**optional*) - Target network updater (typically SoftUpdate).
- **async_collection** (*bool**,**optional*) - Whether to use async data collection. Defaults to False.
- **log_timings** (*bool**,**optional*) - Whether to log timing information for hooks. Defaults to False.
- **done_key** (*NestedKey**,**optional*) - Done key used by losses and logging. Defaults to "done".
- **terminated_key** (*NestedKey**,**optional*) - Terminated key used by losses and logging. Defaults to "terminated".
- **reward_key** (*NestedKey**,**optional*) - Reward key used by losses and logging. Defaults to "reward".
- **episode_reward_key** (*NestedKey**,**optional*) - Episode reward key used for cumulative reward logging.
Defaults to "reward_sum".
- **action_key** (*NestedKey**,**optional*) - Action key used by losses and logging. Defaults to "action".
- **observation_key** (*NestedKey**,**optional*) - Observation key used for logging. Defaults to "observation".

Note

This is an experimental/prototype feature. The API may change in future versions.
DDPG is designed for continuous action spaces. For discrete actions, use DQNTrainer.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function.
They are ignored when `CKPT_BACKEND=memmap`.

Note

When `CKPT_BACKEND=torch`, `weights_only=True` is set by
default for safer deserialization. Pass `weights_only=False`
explicitly only if you have custom (non-stdlib) objects in your
state dict.

Note

When `CKPT_BACKEND=torch`, `mmap=True` is set by default so
the checkpoint is memory-mapped rather than materialized in RAM
at load time. Pass `mmap=False` if the checkpoint was saved
with the legacy (pre-zipfile) `torch.save` format or if
`file` is a file-like object rather than a path.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.