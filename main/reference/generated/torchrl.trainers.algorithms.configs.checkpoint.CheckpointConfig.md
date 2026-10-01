# torchrl.trainers.algorithms.configs.checkpoint.CheckpointConfig

*class*torchrl.trainers.algorithms.configs.checkpoint.CheckpointConfig(*format: str = 'directory'*, *strict: str = 'error'*, *archive_compression: str = 'stored'*, *save_components: list[str] | None = None*, *_target_: str = 'torchrl.checkpoint.Checkpoint'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/checkpoint.html#CheckpointConfig)

Hydra configuration for [`Checkpoint`](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint).

Every kwarg accepted by `Checkpoint.__init__` is exposed as a field here.
Components are registered by the trainer that receives the checkpoint.

See also

[`Checkpoint`](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)