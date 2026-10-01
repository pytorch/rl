# torchrl.trainers.algorithms.configs.utils.AdamConfig

*class*torchrl.trainers.algorithms.configs.utils.AdamConfig(*lr: float = 0.001*, *betas: tuple[float, float] = (0.9, 0.999)*, *eps: float = 0.0001*, *weight_decay: float = 0.0*, *amsgrad: bool = False*, *foreach: bool | None = None*, *maximize: bool = False*, *capturable: bool = False*, *differentiable: bool = False*, *fused: bool | None = None*, *decoupled_weight_decay: bool = False*, *_target_: str = 'torch.optim.Adam'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#AdamConfig)

Hydra configuration for [`torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam).

Every kwarg accepted by `torch.optim.Adam.__init__` is exposed as a field here.