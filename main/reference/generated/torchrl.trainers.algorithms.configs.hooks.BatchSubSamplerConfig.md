# torchrl.trainers.algorithms.configs.hooks.BatchSubSamplerConfig

*class*torchrl.trainers.algorithms.configs.hooks.BatchSubSamplerConfig(*batch_size: int = '???'*, *sub_traj_len: int = 0*, *min_sub_traj_len: int = 0*, *_target_: str = 'torchrl.trainers.trainers.BatchSubSampler'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#BatchSubSamplerConfig)

Configuration for the [`BatchSubSampler`](torchrl.trainers.BatchSubSampler.html#torchrl.trainers.BatchSubSampler) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import BatchSubSamplerConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(BatchSubSamplerConfig(batch_size=64, sub_traj_len=8))
```