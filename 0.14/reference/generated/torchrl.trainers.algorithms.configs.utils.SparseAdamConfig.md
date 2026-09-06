# torchrl.trainers.algorithms.configs.utils.SparseAdamConfig

*class*torchrl.trainers.algorithms.configs.utils.SparseAdamConfig(*lr: float = 0.001*, *betas: tuple[float, float] = (0.9, 0.999)*, *eps: float = 1e-08*, *maximize: bool = False*, *_target_: str = 'torch.optim.SparseAdam'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#SparseAdamConfig)

Hydra configuration for [`torch.optim.SparseAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam).

Every kwarg accepted by `torch.optim.SparseAdam.__init__` is exposed as a field here.