# torchrl.trainers.algorithms.configs.objectives.GAEConfig

*class*torchrl.trainers.algorithms.configs.objectives.GAEConfig(*_partial_: bool = False*, *gamma: float | None = None*, *lmbda: float | None = None*, *value_network: Any = None*, *average_gae: bool = True*, *differentiable: bool = False*, *vectorized: bool | None = None*, *skip_existing: bool | None = None*, *advantage_key: str | None = None*, *value_target_key: str | None = None*, *value_key: str | None = None*, *shifted: bool = False*, *device: Any = None*, *time_dim: int | None = None*, *auto_reset_env: bool = False*, *deactivate_vmap: bool = False*, *value_chunk_size: int | None = None*, *num_chunks: int | None = None*, *num_chunk: int | None = None*, *value_chunk_dim: int = 0*, *shifted_budget: int = 1*, *_target_: str = 'torchrl.objectives.value.GAE'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#GAEConfig)

Hydra configuration for [`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE).

Every kwarg accepted by `GAE.__init__` is exposed as a field here.