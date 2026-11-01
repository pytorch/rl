# torchrl.trainers.algorithms.configs.checkpoint.CheckpointRotationConfig

*class*torchrl.trainers.algorithms.configs.checkpoint.CheckpointRotationConfig(*directory: str*, *keep_last: int*, *keep_best: list[str] | None = None*, *prefix: str = 'checkpoint'*, *_target_: str = 'torchrl.checkpoint.CheckpointRotation'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/checkpoint.html#CheckpointRotationConfig)

Hydra configuration for [`CheckpointRotation`](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation).

`keep_best` is a two-item list `[metadata_key, mode]` with `mode` one
of `"min"` or `"max"`.

See also

[`CheckpointRotation`](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation)