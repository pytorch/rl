# torchrl.trainers.algorithms.configs.modules.AdditiveGaussianModuleConfig

*class*torchrl.trainers.algorithms.configs.modules.AdditiveGaussianModuleConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *spec: Any = None*, *sigma_init: float = 1.0*, *sigma_end: float = 0.1*, *annealing_num_steps: int = 1000*, *mean: float = 0.0*, *std: float = 1.0*, *action_key: Any = 'action'*, *safe: bool = False*, *device: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_additive_gaussian_module'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#AdditiveGaussianModuleConfig)

A class to configure an AdditiveGaussianModule.

Example

```
>>> cfg = AdditiveGaussianModuleConfig(
... spec=None,
... sigma_init=1.0,
... sigma_end=0.1,
... mean=0.0,
... std=1.0,
... action_key="action",
... )
>>> module = instantiate(cfg)
>>> assert isinstance(module, AdditiveGaussianModule)
```

See also

[`torchrl.modules.AdditiveGaussianModule`](torchrl.modules.AdditiveGaussianModule.html#torchrl.modules.AdditiveGaussianModule)