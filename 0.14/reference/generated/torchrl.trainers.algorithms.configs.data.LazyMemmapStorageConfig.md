# torchrl.trainers.algorithms.configs.data.LazyMemmapStorageConfig

*class*torchrl.trainers.algorithms.configs.data.LazyMemmapStorageConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.data.replay_buffers.LazyMemmapStorage'*, *max_size: int | None = None*, *scratch_dir: Any = None*, *device: Any = 'cpu'*, *ndim: int = 1*, *existsok: bool = False*, *compilable: bool = False*, *shared_init: bool = False*, *auto_cleanup: bool | None = None*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#LazyMemmapStorageConfig)

Hydra configuration for [`LazyMemmapStorage`](torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage).

Every kwarg accepted by `LazyMemmapStorage.__init__` is exposed as a field here.