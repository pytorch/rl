# torchrl.trainers.algorithms.configs.utils.RMSpropConfig

*class*torchrl.trainers.algorithms.configs.utils.RMSpropConfig(*lr: float = 0.01*, *alpha: float = 0.99*, *eps: float = 1e-08*, *weight_decay: float = 0.0*, *momentum: float = 0.0*, *centered: bool = False*, *capturable: bool = False*, *foreach: bool | None = None*, *maximize: bool = False*, *differentiable: bool = False*, *_target_: str = 'torch.optim.RMSprop'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#RMSpropConfig)

Hydra configuration for [`torch.optim.RMSprop`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop).

Every kwarg accepted by `torch.optim.RMSprop.__init__` is exposed as a field here.