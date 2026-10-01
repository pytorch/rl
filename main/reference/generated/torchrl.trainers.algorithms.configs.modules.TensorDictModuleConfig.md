# torchrl.trainers.algorithms.configs.modules.TensorDictModuleConfig

*class*torchrl.trainers.algorithms.configs.modules.TensorDictModuleConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *module: [MLPConfig](torchrl.trainers.algorithms.configs.modules.MLPConfig.html#torchrl.trainers.algorithms.configs.modules.MLPConfig) = '???'*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tensordict_module'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TensorDictModuleConfig)

A class to configure a TensorDictModule.

Example

```
>>> cfg = TensorDictModuleConfig(module=MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32), in_keys=["observation"], out_keys=["action"])
>>> module = instantiate(cfg)
>>> assert isinstance(module, TensorDictModule)
>>> assert module(observation=torch.randn(10, 10)).shape == (10, 10)
```

See also

[`tensordict.nn.TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)