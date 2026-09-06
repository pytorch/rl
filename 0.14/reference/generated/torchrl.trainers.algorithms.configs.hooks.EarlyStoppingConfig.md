# torchrl.trainers.algorithms.configs.hooks.EarlyStoppingConfig

*class*torchrl.trainers.algorithms.configs.hooks.EarlyStoppingConfig(*monitor: Any = 'r_evaluation'*, *mode: str = 'max'*, *min_delta: float = 0.0*, *patience: int = 100000*, *wait_for: int = 1000000*, *check_finite: bool = True*, *_target_: str = 'torchrl.trainers.trainers.EarlyStopping'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#EarlyStoppingConfig)

Configuration for the [`EarlyStopping`](torchrl.trainers.EarlyStopping.html#torchrl.trainers.EarlyStopping) hook.

Examples

```
>>> from torchrl.trainers.algorithms.configs.hooks import EarlyStoppingConfig
>>> from hydra.utils import instantiate
>>> hook = instantiate(
... EarlyStoppingConfig(monitor="r_training", patience=10_000)
... )
```