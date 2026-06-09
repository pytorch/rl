# torchrl.trainers.algorithms.configs.hooks.LogTimingConfig

*class*torchrl.trainers.algorithms.configs.hooks.LogTimingConfig(*prefix: str = 'time'*, *percall: bool = True*, *erase: bool = False*, *_target_: str = 'torchrl.trainers.trainers.LogTiming'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#LogTimingConfig)

Configuration for the `LogTiming` hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import LogTimingConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(LogTimingConfig(prefix="time", percall=True))
```