# torchrl.trainers.algorithms.configs.trainers.FQLTrainerConfig

*class*torchrl.trainers.algorithms.configs.trainers.FQLTrainerConfig(*loss_module: Any*, *optimizer: Any*, *replay_buffer: Any*, *target_net_updater: Any*, *offline_steps: int*, *collector: Any = None*, *total_frames: int = 0*, *device: str | None = None*, *batch_size: int | None = None*, *compile_loss: bool = False*, *logger: Any = None*, *clip_grad_norm: bool = True*, *clip_norm: float | None = None*, *progress_bar: bool = False*, *seed: int | None = None*, *save_trainer_interval: int = 10000*, *log_interval: int = 10000*, *save_trainer_file: Any = None*, *checkpoint: Any = None*, *checkpoint_rotation: Any = None*, *checkpoint_metadata: Any = None*, *log_timings: bool = False*, *auto_log_optim_steps: bool = True*, *hooks: list[Any] | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.trainers.make_fql_trainer'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/trainers.html#FQLTrainerConfig)

Hydra configuration for [`FQLTrainer`](torchrl.trainers.algorithms.FQLTrainer.html#torchrl.trainers.algorithms.FQLTrainer).

Optimizer and target updater configurations should be partials: they receive
the instantiated loss parameters and loss module, respectively.