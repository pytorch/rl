# torchrl.trainers.algorithms.configs.utils.LionConfig

*class*torchrl.trainers.algorithms.configs.utils.LionConfig(*lr: float = 0.0001*, *betas: tuple[float, float] = (0.9, 0.99)*, *weight_decay: float = 0.0*, *_target_: str = 'torch.optim.Lion'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#LionConfig)

Configuration for a Lion optimizer node.

Lion is not available on the torch versions TorchRL currently supports
(there is no `torch.optim.Lion`), so this node cannot be instantiated.
The Config is kept so existing Hydra group references (`optimizer/lion`)
do not vanish without a deprecation cycle.