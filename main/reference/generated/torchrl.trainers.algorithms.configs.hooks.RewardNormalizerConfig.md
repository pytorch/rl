# torchrl.trainers.algorithms.configs.hooks.RewardNormalizerConfig

*class*torchrl.trainers.algorithms.configs.hooks.RewardNormalizerConfig(*decay: float = 0.999*, *scale: float = 1.0*, *eps: float | None = None*, *log_pbar: bool = False*, *reward_key: Any = None*, *_target_: str = 'torchrl.trainers.trainers.RewardNormalizer'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#RewardNormalizerConfig)

Configuration for the [`RewardNormalizer`](torchrl.trainers.RewardNormalizer.html#torchrl.trainers.RewardNormalizer) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import RewardNormalizerConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(RewardNormalizerConfig(decay=0.99, scale=1.0))
```