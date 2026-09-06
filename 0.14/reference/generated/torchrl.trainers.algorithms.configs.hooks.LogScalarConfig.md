# torchrl.trainers.algorithms.configs.hooks.LogScalarConfig

*class*torchrl.trainers.algorithms.configs.hooks.LogScalarConfig(*key: Any = ('next', 'reward')*, *logname: str | None = None*, *log_pbar: bool = False*, *include_std: bool = True*, *reduction: str = 'mean'*, *_target_: str = 'torchrl.trainers.trainers.LogScalar'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#LogScalarConfig)

Configuration for the [`LogScalar`](torchrl.trainers.LogScalar.html#torchrl.trainers.LogScalar) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import LogScalarConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(
... LogScalarConfig(key=["next", "reward"], logname="train_reward")
... )
```