# torchrl.trainers.algorithms.configs.utils.ASGDConfig

*class*torchrl.trainers.algorithms.configs.utils.ASGDConfig(*lr: float = 0.01*, *lambd: float = 0.0001*, *alpha: float = 0.75*, *t0: float = 1000000.0*, *weight_decay: float = 0.0*, *foreach: bool | None = None*, *maximize: bool = False*, *differentiable: bool = False*, *capturable: bool = False*, *_target_: str = 'torch.optim.ASGD'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#ASGDConfig)

Hydra configuration for [`torch.optim.ASGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD).

Every kwarg accepted by `torch.optim.ASGD.__init__` is exposed as a field here.