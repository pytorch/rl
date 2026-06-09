# torchrl.trainers.algorithms.configs.hooks.ClearCudaCacheConfig

*class*torchrl.trainers.algorithms.configs.hooks.ClearCudaCacheConfig(*interval: int = '???'*, *_target_: str = 'torchrl.trainers.trainers.ClearCudaCache'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#ClearCudaCacheConfig)

Configuration for the [`ClearCudaCache`](torchrl.trainers.ClearCudaCache.html#torchrl.trainers.ClearCudaCache) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import ClearCudaCacheConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(ClearCudaCacheConfig(interval=100))
```