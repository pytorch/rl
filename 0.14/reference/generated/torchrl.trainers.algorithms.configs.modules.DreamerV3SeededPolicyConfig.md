# torchrl.trainers.algorithms.configs.modules.DreamerV3SeededPolicyConfig

*class*torchrl.trainers.algorithms.configs.modules.DreamerV3SeededPolicyConfig(*_partial_: bool = False*, *module: Any = '???'*, *seed: int = '???'*, *_target_: str = 'torchrl.modules.DreamerV3SeededPolicy'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#DreamerV3SeededPolicyConfig)

Hydra configuration for `DreamerV3SeededPolicy`.

Examples

```
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3DiscreteActorConfig, DreamerV3SeededPolicyConfig
>>> config = DreamerV3SeededPolicyConfig(
... module=DreamerV3DiscreteActorConfig(in_features=6, out_features=3), seed=7,
... )
>>> policy = instantiate(config)
>>> policy.get_extra_state()
{'seed': 7, 'counter': 0}
```