# torchrl.trainers.algorithms.configs.modules.TanhModuleConfig

*class*torchrl.trainers.algorithms.configs.modules.TanhModuleConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *spec: Any = None*, *low: Any = None*, *high: Any = None*, *clamp: bool = False*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tanh_module'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TanhModuleConfig)

A class to configure a TanhModule.

Example

```
>>> cfg = TanhModuleConfig(in_keys=["action"], out_keys=["action"], low=-1.0, high=1.0)
>>> module = instantiate(cfg)
>>> assert isinstance(module, TanhModule)
```

See also

`torchrl.modules.TanhModule`