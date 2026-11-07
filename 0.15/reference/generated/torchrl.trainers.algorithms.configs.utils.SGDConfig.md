# torchrl.trainers.algorithms.configs.utils.SGDConfig

*class*torchrl.trainers.algorithms.configs.utils.SGDConfig(*lr: float = 0.001*, *momentum: float = 0.0*, *dampening: float = 0.0*, *weight_decay: float = 0.0*, *nesterov: bool = False*, *maximize: bool = False*, *foreach: bool | None = None*, *differentiable: bool = False*, *fused: bool | None = None*, *_target_: str = 'torch.optim.SGD'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#SGDConfig)

Hydra configuration for [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD).

Every kwarg accepted by `torch.optim.SGD.__init__` is exposed as a field here.