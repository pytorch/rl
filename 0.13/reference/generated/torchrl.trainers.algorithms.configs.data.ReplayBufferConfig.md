# torchrl.trainers.algorithms.configs.data.ReplayBufferConfig

*class*torchrl.trainers.algorithms.configs.data.ReplayBufferConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.data.replay_buffers.ReplayBuffer'*, *storage: Any = None*, *sampler: Any = None*, *writer: Any = None*, *collate_fn: Any = None*, *pin_memory: bool = False*, *prefetch: int | None = None*, *transform: Any = None*, *transform_factory: Any = None*, *batch_size: int | None = None*, *dim_extend: int | None = None*, *checkpointer: Any = None*, *generator: Any = None*, *shared: bool = False*, *compilable: bool | None = None*, *delayed_init: bool | None = None*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#ReplayBufferConfig)

Hydra configuration for `ReplayBuffer`.

Every kwarg accepted by `ReplayBuffer.__init__` is exposed as a field here.