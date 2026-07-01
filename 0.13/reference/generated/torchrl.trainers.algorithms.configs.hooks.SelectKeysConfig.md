# torchrl.trainers.algorithms.configs.hooks.SelectKeysConfig

*class*torchrl.trainers.algorithms.configs.hooks.SelectKeysConfig(*keys: list[str] = <factory>*, *_target_: str = 'torchrl.trainers.trainers.SelectKeys'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#SelectKeysConfig)

Configuration for the [`SelectKeys`](torchrl.trainers.SelectKeys.html#torchrl.trainers.SelectKeys) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import SelectKeysConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(SelectKeysConfig(keys=["observation", "action"]))
```