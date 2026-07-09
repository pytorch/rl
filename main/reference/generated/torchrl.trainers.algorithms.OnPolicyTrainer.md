# OnPolicyTrainer

*class*torchrl.trainers.algorithms.OnPolicyTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/on_policy.html#OnPolicyTrainer)

Shared implementation for on-policy trainers (PPO, A2C, REINFORCE).

Warning

This is an experimental/prototype feature. The API may change in future versions.
Please report any issues or feedback to help improve this implementation.

This class hosts the training-loop wiring common to on-policy algorithms:
advantage estimation (GAE by default, registered through
[`ValueEstimatorHook`](torchrl.trainers.ValueEstimatorHook.html#torchrl.trainers.ValueEstimatorHook)), replay-buffer handling,
collector weight synchronization, optional learning-rate scheduling
(through [`LRSchedulerHook`](torchrl.trainers.LRSchedulerHook.html#torchrl.trainers.LRSchedulerHook)) and standard logging
hooks. Concrete algorithms ([`PPOTrainer`](torchrl.trainers.algorithms.PPOTrainer.html#torchrl.trainers.algorithms.PPOTrainer),
[`A2CTrainer`](torchrl.trainers.algorithms.A2CTrainer.html#torchrl.trainers.algorithms.A2CTrainer),
[`ReinforceTrainer`](torchrl.trainers.algorithms.ReinforceTrainer.html#torchrl.trainers.algorithms.ReinforceTrainer)) subclass it and only
override class-level defaults such as the number of epochs per batch.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The data collector for gathering training data.
- **total_frames** (*int*) - Total number of frames to train for.
- **frame_skip** (*int*) - Frame skip value for the environment.
- **optim_steps_per_batch** (*int*) - Number of optimization steps per batch.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - The loss module for computing policy and value losses.
- **optimizer** (*optim.Optimizer**,**optional*) - The optimizer for training.
- **lr_scheduler** (*optim.lr_scheduler.LRScheduler**,**optional*) - Learning-rate scheduler,
stepped once per collected batch via [`LRSchedulerHook`](torchrl.trainers.LRSchedulerHook.html#torchrl.trainers.LRSchedulerHook).
- **logger** (*Logger**,**optional*) - Logger for tracking training metrics.
- **clip_grad_norm** (*bool**,**optional*) - Whether to clip gradient norms. Default: True.
- **clip_norm** (*float**,**optional*) - Maximum gradient norm value.
- **progress_bar** (*bool**,**optional*) - Whether to show a progress bar. Default: True.
- **seed** (*int**,**optional*) - Random seed for reproducibility.
- **save_trainer_interval** (*int**,**optional*) - Interval for saving trainer state. Default: 10000.
- **log_interval** (*int**,**optional*) - Interval for logging metrics. Default: 10000.
- **save_trainer_file** (*str**|**pathlib.Path**,**optional*) - File path for saving trainer state.
- **num_epochs** (*int**,**optional*) - Number of epochs per batch. Defaults to the
algorithm-specific class default (e.g. 4 for PPO, 1 for A2C and REINFORCE).
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - Replay buffer for storing data.
- **batch_size** (*int**,**optional*) - Unused; on-policy sub-batch sizes are driven by
the replay buffer's own `batch_size`. Passing a value emits a warning.
- **gamma** (*float**,**optional*) - Discount factor for GAE. Default: 0.99.
- **lmbda** (*float**,**optional*) - Lambda parameter for GAE. Default: 0.95.
- **enable_logging** (*bool**,**optional*) - Whether to enable logging. Default: True.
- **log_rewards** (*bool**,**optional*) - Whether to log rewards. Default: True.
- **log_actions** (*bool**,**optional*) - Whether to log actions. Default: True.
- **log_observations** (*bool**,**optional*) - Whether to log observations. Default: False.
- **async_collection** (*bool**,**optional*) - Whether to use async collection. Default: False.
- **add_gae** (*bool**,**optional*) - Whether to add GAE computation. Default: True.
- **gae** (*Callable**,**optional*) - Custom GAE module. If None and add_gae is True, a default GAE will be created.
- **weight_update_map** (*dict**[**str**,**str**]**,**optional*) - Mapping from collector destination paths (keys in
collector's weight_sync_schemes) to trainer source paths. Required if collector has
weight_sync_schemes configured. Example: {"policy": "loss_module.actor_network",
"replay_buffer.transforms[0]": "loss_module.critic_network"}
- **log_timings** (*bool**,**optional*) - If True, automatically register a LogTiming hook to log
timing information for all hooks to the logger (e.g., wandb, tensorboard).
Timing metrics will be logged with prefix "time/" (e.g., "time/hook/UpdateWeights").
Default is False.
- **auto_log_optim_steps** (*bool**,**optional*) - If True, log the number of optimization
steps after each optimization loop. Default: True.
- **done_key** (*NestedKey**,**optional*) - Done key used by GAE, losses, and logging. Default: "done".
- **terminated_key** (*NestedKey**,**optional*) - Terminated key used by GAE, losses, and logging.
Default: "terminated".
- **reward_key** (*NestedKey**,**optional*) - Reward key used by GAE, losses, and logging. Default: "reward".
- **episode_reward_key** (*NestedKey**,**optional*) - Episode reward key used for cumulative reward logging.
Default: "reward".
- **action_key** (*NestedKey**,**optional*) - Action key used by losses and logging. Default: "action".
- **observation_key** (*NestedKey**,**optional*) - Observation key used for logging. Default: "observation".

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