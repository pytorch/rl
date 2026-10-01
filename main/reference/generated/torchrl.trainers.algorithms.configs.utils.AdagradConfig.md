# torchrl.trainers.algorithms.configs.utils.AdagradConfig

*class*torchrl.trainers.algorithms.configs.utils.AdagradConfig(*lr: float = 0.01*, *lr_decay: float = 0.0*, *weight_decay: float = 0.0*, *initial_accumulator_value: float = 0.0*, *eps: float = 1e-10*, *foreach: bool | None = None*, *maximize: bool = False*, *differentiable: bool = False*, *fused: bool | None = None*, *_target_: str = 'torch.optim.Adagrad'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#AdagradConfig)

Hydra configuration for [`torch.optim.Adagrad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad).

Every kwarg accepted by `torch.optim.Adagrad.__init__` is exposed as a field here.