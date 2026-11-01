# torchrl.trainers.algorithms.configs.utils.AdadeltaConfig

*class*torchrl.trainers.algorithms.configs.utils.AdadeltaConfig(*lr: float = 1.0*, *rho: float = 0.9*, *eps: float = 1e-06*, *weight_decay: float = 0.0*, *foreach: bool | None = None*, *capturable: bool = False*, *maximize: bool = False*, *differentiable: bool = False*, *_target_: str = 'torch.optim.Adadelta'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#AdadeltaConfig)

Hydra configuration for [`torch.optim.Adadelta`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta).

Every kwarg accepted by `torch.optim.Adadelta.__init__` is exposed as a field here.