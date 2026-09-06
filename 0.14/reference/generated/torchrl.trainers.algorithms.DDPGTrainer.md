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
- **batch_size** (*int**,**optional*) - Global learner batch size. Defaults to the
replay buffer batch size.
- **learner_backend** (*str*) - Optimization placement, `"local"` or `"ray"`.
- **learner_backend_options** (*dict**,**optional*) - Ray world size and resources.
- **learner_poll_interval** (*float*) - Remote replay polling interval.
- **enable_logging** (*bool**,**optional*) - Whether to enable metric logging. Defaults to True.
- **log_rewards** (*bool**,**optional*) - Whether to log reward statistics. Defaults to True.
- **log_actions** (*bool**,**optional*) - Whether to log action statistics. Defaults to True.
- **log_observations** (*bool**,**optional*) - Whether to log observation statistics. Defaults to False.
- **target_net_updater** (*TargetNetUpdater*) - Target network updater (typically
`SoftUpdate`).
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

compute_loss(*sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *method: str | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | tuple[Any, ...]

Evaluate the configured loss through the active execution boundary.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function for
legacy torch checkpoints and unified components explicitly saved with
the torch state-dict payload format. Unified checkpoints additionally
accept `strict` to control missing or incompatible components.
Arguments are ignored when `CKPT_BACKEND=memmap`.

Note

Unified state-dict components use TensorDict storage by default and
do not invoke the pickle loader. For explicit torch payloads and
`CKPT_BACKEND=torch` checkpoints, `weights_only=True` is the
default for safer deserialization. Pass `weights_only=False`
explicitly only if the state dict contains custom objects. On
torch < 2.4 the default is `weights_only=False` because the
weights-only unpickler of those versions cannot deserialize the
`torch.device` instances contained in TensorDict state-dicts.

Note

Explicit torch payloads and `CKPT_BACKEND=torch` checkpoints use
`mmap=True` by default. Pass `mmap=False` for legacy pre-zipfile
`torch.save` files or file-like objects. On Windows the default
is `mmap=False` because a mapped checkpoint keeps the file locked,
preventing deletion or re-save.

Note

Unified checkpoint tensors are mapped to CPU by default. Pass an
explicit `map_location` to select another device mapping.

Note

After restoring an independently registered policy component, the
trainer synchronizes the collector once so local policy copies and
remote workers observe the restored learner weights.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.