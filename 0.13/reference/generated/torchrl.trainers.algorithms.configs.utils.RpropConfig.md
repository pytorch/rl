# torchrl.trainers.algorithms.configs.utils.RpropConfig

*class*torchrl.trainers.algorithms.configs.utils.RpropConfig(*lr: float = 0.01*, *etas: tuple[float, float] = (0.5, 1.2)*, *step_sizes: tuple[float, float] = (1e-06, 50.0)*, *capturable: bool = False*, *foreach: bool | None = None*, *maximize: bool = False*, *differentiable: bool = False*, *_target_: str = 'torch.optim.Rprop'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#RpropConfig)

Hydra configuration for [`torch.optim.Rprop`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop).

Every kwarg accepted by `torch.optim.Rprop.__init__` is exposed as a field here.