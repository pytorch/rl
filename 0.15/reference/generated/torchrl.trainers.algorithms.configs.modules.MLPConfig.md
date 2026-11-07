# torchrl.trainers.algorithms.configs.modules.MLPConfig

*class*torchrl.trainers.algorithms.configs.modules.MLPConfig(*_partial_: bool = False*, *in_features: int | None = None*, *out_features: ~typing.Any = None*, *depth: int | None = None*, *num_cells: ~typing.Any = None*, *activation_class: ~torchrl.trainers.algorithms.configs.modules.ActivationConfig = <factory>*, *activation_kwargs: ~typing.Any = None*, *norm_class: ~typing.Any = None*, *norm_kwargs: ~typing.Any = None*, *dropout: float | None = None*, *bias_last_layer: bool = True*, *single_bias_last_layer: bool = False*, *layer_class: ~torchrl.trainers.algorithms.configs.modules.LayerConfig = <factory>*, *layer_kwargs: dict | None = None*, *activate_last_layer: bool = False*, *device: ~typing.Any = None*, *_target_: str = 'torchrl.modules.MLP'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#MLPConfig)

A class to configure a multi-layer perceptron.

Example

```
>>> cfg = MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32)
>>> net = instantiate(cfg)
>>> y = net(torch.randn(1, 10))
>>> assert y.shape == (1, 5)
```

See also

[`torchrl.modules.MLP`](torchrl.modules.MLP.html#torchrl.modules.MLP)