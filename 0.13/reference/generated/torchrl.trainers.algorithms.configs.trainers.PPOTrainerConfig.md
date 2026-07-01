# torchrl.trainers.algorithms.configs.trainers.PPOTrainerConfig

*class*torchrl.trainers.algorithms.configs.trainers.PPOTrainerConfig(*collector: Any*, *total_frames: int*, *optim_steps_per_batch: int | None*, *loss_module: Any*, *optimizer: Any*, *logger: Any*, *save_trainer_file: Any*, *replay_buffer: Any*, *frame_skip: int = 1*, *clip_grad_norm: bool = True*, *clip_norm: float | None = None*, *progress_bar: bool = True*, *seed: int | None = None*, *save_trainer_interval: int = 10000*, *log_interval: int = 10000*, *create_env_fn: Any = None*, *actor_network: Any = None*, *critic_network: Any = None*, *num_epochs: int = 4*, *async_collection: bool = False*, *add_gae: bool = True*, *gae: Any = None*, *weight_update_map: dict[str, str] | None = None*, *log_timings: bool = False*, *auto_log_optim_steps: bool = True*, *batch_size: int | None = None*, *gamma: float = 0.99*, *lmbda: float = 0.95*, *enable_logging: bool = True*, *log_rewards: bool = True*, *log_actions: bool = True*, *log_observations: bool = False*, *done_key: Any = 'done'*, *terminated_key: Any = 'terminated'*, *reward_key: Any = 'reward'*, *episode_reward_key: Any = 'reward'*, *action_key: Any = 'action'*, *observation_key: Any = 'observation'*, *hooks: list[Any] | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/trainers.html#PPOTrainerConfig)

Hydra configuration for [`PPOTrainer`](torchrl.trainers.algorithms.PPOTrainer.html#torchrl.trainers.algorithms.PPOTrainer).

Every kwarg accepted by `PPOTrainer.__init__` is exposed as a field here.

Parameters:

- **collector** - The data collector for gathering training data.
- **total_frames** - Total number of frames to train for.
- **optim_steps_per_batch** - Number of optimization steps per batch.
- **loss_module** - The loss module for computing policy and value losses.
- **optimizer** - The optimizer for training.
- **logger** - Logger for tracking training metrics.
- **save_trainer_file** - File path for saving trainer state.
- **replay_buffer** - Replay buffer for storing data.
- **frame_skip** - Frame skip value for the environment. Default: 1.
- **clip_grad_norm** - Whether to clip gradient norms. Default: True.
- **clip_norm** - Maximum gradient norm value.
- **progress_bar** - Whether to show a progress bar. Default: True.
- **seed** - Random seed for reproducibility.
- **save_trainer_interval** - Interval for saving trainer state. Default: 10000.
- **log_interval** - Interval for logging metrics. Default: 10000.
- **create_env_fn** - Environment creation function.
- **actor_network** - Actor network configuration.
- **critic_network** - Critic network configuration.
- **num_epochs** - Number of epochs per batch. Default: 4.
- **async_collection** - Whether to use async collection. Default: False.
- **add_gae** - Whether to add GAE computation. Default: True.
- **gae** - Custom GAE module configuration.
- **weight_update_map** - Mapping from collector destination paths to trainer source paths.
Required if collector has weight_sync_schemes configured.
Example: `{"policy": "loss_module.actor_network", "replay_buffer.transforms[0]": "loss_module.critic_network"}`.
- **log_timings** - Whether to automatically log timing information for all hooks.
If True, timing metrics will be logged to the logger (e.g., wandb, tensorboard)
with prefix "time/" (e.g., "time/hook/UpdateWeights"). Default: False.