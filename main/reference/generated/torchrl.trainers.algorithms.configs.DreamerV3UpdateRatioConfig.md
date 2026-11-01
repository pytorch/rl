# DreamerV3UpdateRatioConfig

*class*torchrl.trainers.algorithms.configs.DreamerV3UpdateRatioConfig(*ratio: float = '???'*, *_target_: str = 'torchrl.trainers.algorithms.DreamerV3UpdateRatio'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#DreamerV3UpdateRatioConfig)

Hydra configuration for [`DreamerV3UpdateRatio`](torchrl.trainers.algorithms.DreamerV3UpdateRatio.html#torchrl.trainers.algorithms.DreamerV3UpdateRatio).

Examples

```
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3UpdateRatioConfig
>>> schedule = instantiate(DreamerV3UpdateRatioConfig(ratio=0.25))
>>> schedule(4), schedule(8)
(1, 1)
```