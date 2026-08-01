# torchrl.trainers.algorithms.configs.hooks.CountFramesLogConfig

*class*torchrl.trainers.algorithms.configs.hooks.CountFramesLogConfig(*frame_skip: int = 1*, *log_pbar: bool = False*, *_target_: str = 'torchrl.trainers.trainers.CountFramesLog'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#CountFramesLogConfig)

Configuration for the [`CountFramesLog`](torchrl.trainers.CountFramesLog.html#torchrl.trainers.CountFramesLog) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import CountFramesLogConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(CountFramesLogConfig(frame_skip=4))
```