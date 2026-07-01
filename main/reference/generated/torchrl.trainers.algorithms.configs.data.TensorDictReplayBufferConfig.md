# torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig

*class*torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.data.replay_buffers.TensorDictReplayBuffer'*, *priority_key: str = 'td_error'*, *sampler: Any = None*, *storage: Any = None*, *writer: Any = None*, *collate_fn: Any = None*, *pin_memory: bool = False*, *prefetch: int | None = None*, *transform: Any = None*, *transform_factory: Any = None*, *batch_size: int | None = None*, *dim_extend: int | None = None*, *checkpointer: Any = None*, *generator: Any = None*, *consume_after_n_samples: int | None = None*, *shared: bool = False*, *compilable: bool | None = None*, *delayed_init: bool | None = None*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#TensorDictReplayBufferConfig)

Hydra configuration for `TensorDictReplayBuffer`.

Every kwarg accepted by `TensorDictReplayBuffer.__init__` (plus the `ReplayBuffer`
kwargs it forwards via `**kwargs`) is exposed as a field here.