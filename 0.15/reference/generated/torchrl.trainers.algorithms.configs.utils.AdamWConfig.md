# torchrl.trainers.algorithms.configs.utils.AdamWConfig

*class*torchrl.trainers.algorithms.configs.utils.AdamWConfig(*lr: float = 0.001*, *betas: tuple[float, float] = (0.9, 0.999)*, *eps: float = 1e-08*, *weight_decay: float = 0.01*, *amsgrad: bool = False*, *maximize: bool = False*, *foreach: bool | None = None*, *capturable: bool = False*, *differentiable: bool = False*, *fused: bool | None = None*, *_target_: str = 'torch.optim.AdamW'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#AdamWConfig)

Hydra configuration for [`torch.optim.AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW).

Every kwarg accepted by `torch.optim.AdamW.__init__` is exposed as a field here.