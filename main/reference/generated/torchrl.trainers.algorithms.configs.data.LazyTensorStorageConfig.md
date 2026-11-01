# torchrl.trainers.algorithms.configs.data.LazyTensorStorageConfig

*class*torchrl.trainers.algorithms.configs.data.LazyTensorStorageConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.data.replay_buffers.LazyTensorStorage'*, *max_size: int | None = None*, *device: Any = 'cpu'*, *ndim: int = 1*, *compilable: bool = False*, *consolidated: bool = False*, *shared_init: bool = False*, *cleanup_memmap: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#LazyTensorStorageConfig)

Hydra configuration for [`LazyTensorStorage`](torchrl.data.replay_buffers.LazyTensorStorage.html#torchrl.data.replay_buffers.LazyTensorStorage).

Every kwarg accepted by `LazyTensorStorage.__init__` is exposed as a field here.