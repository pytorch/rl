# torchrl.trainers.algorithms.configs.modules.ConvNetConfig

*class*torchrl.trainers.algorithms.configs.modules.ConvNetConfig(*_partial_: bool = False*, *in_features: int | None = None*, *depth: int | None = None*, *num_cells: ~typing.Any = None*, *kernel_sizes: ~typing.Any = 3*, *strides: ~typing.Any = 1*, *paddings: ~typing.Any = 0*, *activation_class: ~torchrl.trainers.algorithms.configs.modules.ActivationConfig = <factory>*, *activation_kwargs: ~typing.Any = None*, *norm_class: ~torchrl.trainers.algorithms.configs.modules.NormConfig | None = None*, *norm_kwargs: ~typing.Any = None*, *bias_last_layer: bool = True*, *aggregator_class: ~torchrl.trainers.algorithms.configs.modules.AggregatorConfig = <factory>*, *aggregator_kwargs: dict | None = None*, *squeeze_output: bool = False*, *device: ~typing.Any = None*, *_target_: str = 'torchrl.modules.ConvNet'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#ConvNetConfig)

A class to configure a convolutional network.

Defaults to [`torchrl.modules.ConvNet`](torchrl.modules.ConvNet.html#torchrl.modules.ConvNet).

Example

```
>>> cfg = ConvNetConfig(in_features=3, depth=2, num_cells=[32, 64], kernel_sizes=[3, 5], strides=[1, 2], paddings=[1, 2])
>>> net = instantiate(cfg)
>>> y = net(torch.randn(1, 3, 32, 32))
>>> assert y.shape == (1, 64)
```

See also

[`torchrl.modules.ConvNet`](torchrl.modules.ConvNet.html#torchrl.modules.ConvNet)