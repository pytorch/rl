# DdpgCnnQNet

*class*torchrl.modules.DdpgCnnQNet(*conv_net_kwargs: dict | None = None*, *mlp_net_kwargs: dict | None = None*, *use_avg_pooling: bool = True*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgCnnQNet)

DDPG Convolutional Q-value class.

Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
[https://arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf)

The DDPG Q-value network takes as input an observation and an action, and
returns a scalar from it.

Parameters:

- **conv_net_kwargs** (*dict**,**optional*) -

kwargs for the
convolutional network.
Defaults to

```
>>> {
... 'in_features': None,
... "num_cells": [32, 64, 128],
... "kernel_sizes": [8, 4, 3],
... "strides": [4, 2, 1],
... "paddings": [0, 0, 1],
... 'activation_class': nn.ELU,
... 'norm_class': None,
... 'aggregator_class': nn.AdaptiveAvgPool2d,
... 'aggregator_kwargs': {},
... 'squeeze_output': True,
... }
```
- **mlp_net_kwargs** (*dict**,**optional*) -

kwargs for MLP.
Defaults to

```
>>> {
... 'in_features': None,
... 'out_features': 1,
... 'depth': 2,
... 'num_cells': 200,
... 'activation_class': nn.ELU,
... 'bias_last_layer': True,
... }
```
- **use_avg_pooling** (*bool**,**optional*) - if `True`, a
`AvgPooling` layer is used to aggregate the
output. Default is `True`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> from torchrl.modules import DdpgCnnQNet
>>> import torch
>>> net = DdpgCnnQNet()
>>> print(net)
DdpgCnnQNet(
 (convnet): ConvNet(
 (0): LazyConv2d(0, 32, kernel_size=(8, 8), stride=(4, 4))
 (1): ELU(alpha=1.0)
 (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
 (3): ELU(alpha=1.0)
 (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (5): ELU(alpha=1.0)
 (6): AdaptiveAvgPool2d(output_size=(1, 1))
 (7): Squeeze2dLayer()
 )
 (mlp): MLP(
 (0): LazyLinear(in_features=0, out_features=200, bias=True)
 (1): ELU(alpha=1.0)
 (2): Linear(in_features=200, out_features=200, bias=True)
 (3): ELU(alpha=1.0)
 (4): Linear(in_features=200, out_features=1, bias=True)
 )
)
>>> obs = torch.zeros(1, 3, 64, 64)
>>> action = torch.zeros(1, 4)
>>> value = net(obs, action)
>>> print(value.shape)
torch.Size([1, 1])
```

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgCnnQNet.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.