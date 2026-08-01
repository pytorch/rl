# torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig

*class*torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.data.replay_buffers.TensorDictReplayBuffer'*, *priority_key: str = 'td_error'*, *sampler: ~typing.Any = None*, *storage: ~typing.Any = None*, *writer: ~typing.Any = None*, *collate_fn: ~typing.Any = None*, *pin_memory: bool = False*, *prefetch: int | None = None*, *transform: ~typing.Any = None*, *transform_factory: ~typing.Any = None*, *batch_size: int | None = None*, *dim_extend: int | None = None*, *checkpointer: ~typing.Any = None*, *generator: ~typing.Any = None*, *consume_after_n_samples: int | None = None*, *shared: bool = False*, *compilable: bool | None = None*, *delayed_init: bool | None = None*, *service_backend: str = 'direct'*, *service_backend_options: dict[str*, *~typing.Any] = <factory>*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#TensorDictReplayBufferConfig)

Hydra configuration for `TensorDictReplayBuffer`.

Every kwarg accepted by `TensorDictReplayBuffer.__init__` (plus the `ReplayBuffer`
kwargs it forwards via `**kwargs`) is exposed as a field here.