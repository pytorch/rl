# PPOTrainer

*class*torchrl.trainers.algorithms.PPOTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/ppo.html#PPOTrainer)

PPO (Proximal Policy Optimization) trainer implementation.

See also `PPOTrainerConfig` for the
Hydra configuration counterpart.

Warning

This is an experimental/prototype feature. The API may change in future versions.
Please report any issues or feedback to help improve this implementation.

This trainer implements the PPO algorithm for training reinforcement learning agents.
It extends the base Trainer class with PPO-specific functionality including
policy optimization, value function learning, and entropy regularization.

PPO typically uses multiple epochs of optimization on the same batch of data.
This trainer defaults to 4 epochs, which is a common choice for PPO implementations.

The trainer includes comprehensive logging capabilities for monitoring training progress:
- Training rewards (mean, std, max, total)
- Action statistics (norms)
- Episode completion rates
- Observation statistics (optional)

Logging can be configured via constructor parameters to enable/disable specific metrics.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The data collector for gathering training data.
- **total_frames** (*int*) - Total number of frames to train for.
- **frame_skip** (*int*) - Frame skip value for the environment.
- **optim_steps_per_batch** (*int*) - Number of optimization steps per batch.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - The loss module for computing policy and value losses.
- **optimizer** (*optim.Optimizer**,**optional*) - The optimizer for training.
- **logger** (*Logger**,**optional*) - Logger for tracking training metrics.
- **clip_grad_norm** (*bool**,**optional*) - Whether to clip gradient norms. Default: True.
- **clip_norm** (*float**,**optional*) - Maximum gradient norm value.
- **progress_bar** (*bool**,**optional*) - Whether to show a progress bar. Default: True.
- **seed** (*int**,**optional*) - Random seed for reproducibility.
- **save_trainer_interval** (*int**,**optional*) - Interval for saving trainer state. Default: 10000.
- **log_interval** (*int**,**optional*) - Interval for logging metrics. Default: 10000.
- **save_trainer_file** (*str**|**pathlib.Path**,**optional*) - File path for saving trainer state.
- **num_epochs** (*int**,**optional*) - Number of epochs per batch. Default: 4.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - Replay buffer for storing data.
- **batch_size** (*int**,**optional*) - Batch size for optimization.
- **gamma** (*float**,**optional*) - Discount factor for GAE. Default: 0.9.
- **lmbda** (*float**,**optional*) - Lambda parameter for GAE. Default: 0.99.
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
- **done_key** (*NestedKey**,**optional*) - Done key used by GAE, losses, and logging. Default: "done".
- **terminated_key** (*NestedKey**,**optional*) - Terminated key used by GAE, losses, and logging.
Default: "terminated".
- **reward_key** (*NestedKey**,**optional*) - Reward key used by GAE, losses, and logging. Default: "reward".
- **episode_reward_key** (*NestedKey**,**optional*) - Episode reward key used for cumulative reward logging.
Default: "reward".
- **action_key** (*NestedKey**,**optional*) - Action key used by losses and logging. Default: "action".
- **observation_key** (*NestedKey**,**optional*) - Observation key used for logging. Default: "observation".

Examples

```
>>> # Basic usage with manual configuration
>>> from torchrl.trainers.algorithms.ppo import PPOTrainer
>>> from torchrl.trainers.algorithms.configs import PPOTrainerConfig
>>> from hydra import instantiate
>>> config = PPOTrainerConfig(...) # Configure with required parameters
>>> trainer = instantiate(config)
>>> trainer.train()
```

Note

This trainer requires a configurable environment setup. See the
`configs` module for configuration options.

Warning

This is an experimental feature. The API may change in future versions.
We welcome feedback and contributions to help improve this implementation!

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function.
They are ignored when `CKPT_BACKEND=memmap`.

Note

When `CKPT_BACKEND=torch`, `weights_only=True` is set by
default for safer deserialization. Pass `weights_only=False`
explicitly only if you have custom (non-stdlib) objects in your
state dict.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.