# torchrl.trainers.algorithms.configs.data.ListStorageConfig

*class*torchrl.trainers.algorithms.configs.data.ListStorageConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.data.replay_buffers.ListStorage'*, *max_size: int | None = None*, *compilable: bool = False*, *device: Any = None*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#ListStorageConfig)

Hydra configuration for [`ListStorage`](torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage).

Every kwarg accepted by `ListStorage.__init__` is exposed as a field here.