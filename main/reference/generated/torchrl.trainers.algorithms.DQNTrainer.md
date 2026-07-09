# DQNTrainer

*class*torchrl.trainers.algorithms.DQNTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/dqn.html#DQNTrainer)

A trainer class for Deep Q-Network (DQN) algorithm.

See also `DQNTrainerConfig` for the
Hydra configuration counterpart.

This trainer implements the DQN algorithm, a value-based method for discrete
action spaces that learns a Q-function and derives a greedy policy from it.

The trainer handles:
- Replay buffer management for off-policy learning
- Target network updates (typically HardUpdate) with configurable update frequency
- Policy weight updates to the data collector
- Comprehensive logging of training metrics

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The data collector used to gather environment interactions.
- **total_frames** (*int*) - Total number of frames to collect during training.
- **frame_skip** (*int*) - Number of frames to skip between policy updates.
- **optim_steps_per_batch** (*int*) - Number of optimization steps per collected batch.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)*|**Callable*) - The DQN loss module or a callable that computes losses.
- **optimizer** (*optim.Optimizer**,**optional*) - The optimizer for training.
- **logger** (*Logger**,**optional*) - Logger for recording training metrics. Defaults to None.
- **clip_grad_norm** (*bool**,**optional*) - Whether to clip gradient norms. Defaults to True.
- **clip_norm** (*float**,**optional*) - Maximum gradient norm for clipping. Defaults to None.
- **progress_bar** (*bool**,**optional*) - Whether to show a progress bar during training. Defaults to True.
- **seed** (*int**,**optional*) - Random seed for reproducibility. Defaults to None.
- **save_trainer_interval** (*int**,**optional*) - Interval for saving trainer state. Defaults to 10000.
- **log_interval** (*int**,**optional*) - Interval for logging metrics. Defaults to 10000.
- **save_trainer_file** (*str**|**pathlib.Path**,**optional*) - File path for saving trainer state. Defaults to None.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - Replay buffer for storing and sampling experiences. Defaults to None.
- **enable_logging** (*bool**,**optional*) - Whether to enable metric logging. Defaults to True.
- **log_rewards** (*bool**,**optional*) - Whether to log reward statistics. Defaults to True.
- **log_observations** (*bool**,**optional*) - Whether to log observation statistics. Defaults to False.
- **target_net_updater** (*TargetNetUpdater**,**optional*) - Target network updater (typically HardUpdate). Defaults to None.
- **greedy_module** ([*EGreedyModule*](torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule)*,**optional*) - Epsilon-greedy exploration module. When provided,
the module's epsilon is annealed during training. Defaults to None.
- **async_collection** (*bool**,**optional*) - Whether to use async data collection. Defaults to False.
- **log_timings** (*bool**,**optional*) - Whether to log timing information for hooks. Defaults to False.
- **mixing_strategy** (*str**,**optional*) - Multi-agent mixing strategy. Accepted values are `"qmix"` and
`"vdn"` for mixed-value training, `"iql"` for independent Q-learning, or None for standard
DQN. Defaults to None.
- **done_key** (*NestedKey**,**optional*) - Key for the done signal used by logging. Defaults to `"done"`.
- **terminated_key** (*NestedKey**,**optional*) - Key for the terminated signal. Defaults to `"terminated"`.
- **reward_key** (*NestedKey**,**optional*) - Source reward key used by logging and reward aggregation.
Defaults to `"reward"`.
- **episode_reward_key** (*NestedKey**,**optional*) - Source episode reward key used by logging and reward
aggregation. Defaults to `"reward_sum"`.
- **aggregated_reward_key** (*NestedKey**,**optional*) - Destination key for rewards averaged over the agent
dimension when using QMIX or VDN. The source is `reward_key`. Set this to `reward_key` to
overwrite the source reward in-place. Required when `mixing_strategy` is `"qmix"` or
`"vdn"`. Defaults to None.
- **aggregated_episode_reward_key** (*NestedKey**,**optional*) - Destination key for episode rewards averaged over
the agent dimension when using QMIX or VDN. The source is `episode_reward_key`. Set this to
`episode_reward_key` to overwrite the source reward in-place. Required when `mixing_strategy`
is `"qmix"` or `"vdn"`. Defaults to None.
- **action_key** (*NestedKey**,**optional*) - Key for actions used by the exploration module and policy specs.
Defaults to `"action"`.
- **observation_key** (*NestedKey**,**optional*) - Key for observations used by logging. Defaults to
`"observation"`.

Example

```
>>> from torchrl.collectors import Collector
>>> from torchrl.objectives import DQNLoss
>>> from torchrl.data import ReplayBuffer, LazyTensorStorage
>>> from torchrl.objectives.utils import HardUpdate
>>> from torch import optim
>>>
>>> # Set up collector, loss, and replay buffer
>>> collector = Collector(env, policy, frames_per_batch=128)
>>> loss_module = DQNLoss(value_network, delay_value=True)
>>> optimizer = optim.Adam(loss_module.parameters(), lr=2.5e-4)
>>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(100000))
>>> target_net_updater = HardUpdate(loss_module, value_network_update_interval=50)
>>>
>>> trainer = DQNTrainer(
... collector=collector,
... total_frames=500000,
... frame_skip=1,
... optim_steps_per_batch=10,
... loss_module=loss_module,
... optimizer=optimizer,
... replay_buffer=replay_buffer,
... target_net_updater=target_net_updater,
... )
>>> trainer.train()
```

Note

This is an experimental/prototype feature. The API may change in future versions.
DQN is designed for discrete action spaces (e.g., CartPole, Atari).
For continuous control, consider using SACTrainer or DDPGTrainer instead.

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