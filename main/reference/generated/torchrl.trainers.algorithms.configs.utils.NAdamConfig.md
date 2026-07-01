# torchrl.trainers.algorithms.configs.utils.NAdamConfig

*class*torchrl.trainers.algorithms.configs.utils.NAdamConfig(*lr: float = 0.002*, *betas: tuple[float, float] = (0.9, 0.999)*, *eps: float = 1e-08*, *weight_decay: float = 0.0*, *momentum_decay: float = 0.004*, *decoupled_weight_decay: bool = False*, *foreach: bool | None = None*, *maximize: bool = False*, *capturable: bool = False*, *differentiable: bool = False*, *_target_: str = 'torch.optim.NAdam'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#NAdamConfig)

Hydra configuration for [`torch.optim.NAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam).

Every kwarg accepted by `torch.optim.NAdam.__init__` is exposed as a field here.