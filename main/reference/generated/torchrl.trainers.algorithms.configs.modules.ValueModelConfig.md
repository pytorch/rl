# torchrl.trainers.algorithms.configs.modules.ValueModelConfig

*class*torchrl.trainers.algorithms.configs.modules.ValueModelConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_value_model'*, *network: [NetworkConfig](torchrl.trainers.algorithms.configs.modules.NetworkConfig.html#torchrl.trainers.algorithms.configs.modules.NetworkConfig) = '???'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#ValueModelConfig)

A class to configure a Value model.

Example

```
>>> cfg = ValueModelConfig(network=MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32))
>>> net = instantiate(cfg)
>>> y = net(torch.randn(1, 10))
>>> assert y.shape == (1, 5)
```

See also

[`torchrl.modules.ValueOperator`](torchrl.modules.ValueOperator.html#torchrl.modules.ValueOperator)