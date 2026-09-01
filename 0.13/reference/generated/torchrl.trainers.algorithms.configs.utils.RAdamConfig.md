# torchrl.trainers.algorithms.configs.utils.RAdamConfig

*class*torchrl.trainers.algorithms.configs.utils.RAdamConfig(*lr: float = 0.001*, *betas: tuple[float, float] = (0.9, 0.999)*, *eps: float = 1e-08*, *weight_decay: float = 0.0*, *decoupled_weight_decay: bool = False*, *foreach: bool | None = None*, *maximize: bool = False*, *capturable: bool = False*, *differentiable: bool = False*, *_target_: str = 'torch.optim.RAdam'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#RAdamConfig)

Hydra configuration for [`torch.optim.RAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam).

Every kwarg accepted by `torch.optim.RAdam.__init__` is exposed as a field here.