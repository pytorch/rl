# TD3Trainer

*class*torchrl.trainers.algorithms.TD3Trainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/td3.html#TD3Trainer)

A trainer class for Twin Delayed DDPG (TD3) algorithm.

See also `TD3TrainerConfig` for the
Hydra configuration counterpart.

This trainer implements the TD3 algorithm, an off-policy actor-critic method
that builds on DDPG with improvements for stability including:
- Clipped double Q-learning
- Delayed policy updates
- Target policy smoothing

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
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)*|**Callable*) - The TD3 loss module or a callable that computes losses.
- **optimizer** (*optim.Optimizer**,**optional*) - Fallback optimizer for training. Defaults to None.
- **optimization_stepper** (*TD3OptimizationStepper**,**optional*) - Custom optimization stepper
controlling delayed actor/critic updates. Defaults to None.
- **logger** (*Logger**,**optional*) - Logger for recording training metrics. Defaults to None.
- **clip_grad_norm** (*bool**,**optional*) - Whether to clip gradient norms. Defaults to True.
- **clip_norm** (*float**,**optional*) - Maximum gradient norm for clipping. Defaults to None.
- **progress_bar** (*bool**,**optional*) - Whether to show a progress bar during training. Defaults to True.
- **seed** (*int**,**optional*) - Random seed for reproducibility. Defaults to None.
- **save_trainer_interval** (*int**,**optional*) - Interval for saving trainer state. Defaults to 10000.
- **log_interval** (*int**,**optional*) - Interval for logging metrics. Defaults to 10000.
- **save_trainer_file** (*str**|**pathlib.Path**,**optional*) - File path for saving trainer state. Defaults to None.
- **num_epochs** (*int**,**optional*) - Number of epochs per batch. Defaults to 1 (typical for off-policy).
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - Replay buffer for storing and sampling experiences. Defaults to None.
- **enable_logging** (*bool**,**optional*) - Whether to enable metric logging. Defaults to True.
- **log_rewards** (*bool**,**optional*) - Whether to log reward statistics. Defaults to True.
- **log_actions** (*bool**,**optional*) - Whether to log action statistics. Defaults to True.
- **log_observations** (*bool**,**optional*) - Whether to log observation statistics. Defaults to False.
- **async_collection** (*bool**,**optional*) - Whether to use async collection. Defaults to False.
- **log_timings** (*bool**,**optional*) - Whether to log timing information. Defaults to False.
- **target_net_updater** (*TargetNetUpdater*) - Target network updater (typically SoftUpdate).
- **exploration_module** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*,**optional*) - Optional exploration module appended
to actor weights when syncing policy parameters to the collector. Defaults to None.

Note

This is an experimental/prototype feature. The API may change in future versions.
TD3 is particularly effective for continuous control tasks.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function.
They are ignored when `CKPT_BACKEND=memmap`.

Note

When `CKPT_BACKEND=torch`, `weights_only=True` is set by
default for safer deserialization. Pass `weights_only=False`
explicitly only if you have custom (non-stdlib) objects in your
state dict. On torch < 2.4 the default is `weights_only=False`
because the weights-only unpickler of those versions cannot
deserialize the `torch.device` instances contained in
TensorDict state-dicts.

Note

When `CKPT_BACKEND=torch`, `mmap=True` is set by default so
the checkpoint is memory-mapped rather than materialized in RAM
at load time. Pass `mmap=False` if the checkpoint was saved
with the legacy (pre-zipfile) `torch.save` format or if
`file` is a file-like object rather than a path. On Windows
the default is `mmap=False`: a mapped checkpoint would keep
the file locked, preventing it from being deleted or re-saved
while the loaded state is alive.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.