# torchrl.trainers.algorithms.configs.utils.AdamaxConfig

*class*torchrl.trainers.algorithms.configs.utils.AdamaxConfig(*lr: float = 0.002*, *betas: tuple[float, float] = (0.9, 0.999)*, *eps: float = 1e-08*, *weight_decay: float = 0.0*, *foreach: bool | None = None*, *maximize: bool = False*, *differentiable: bool = False*, *capturable: bool = False*, *_target_: str = 'torch.optim.Adamax'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#AdamaxConfig)

Hydra configuration for [`torch.optim.Adamax`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax).

Every kwarg accepted by `torch.optim.Adamax.__init__` is exposed as a field here.