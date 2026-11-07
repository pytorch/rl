# DdpgCnnActor

*class*torchrl.modules.DdpgCnnActor(*action_dim: int*, *conv_net_kwargs: dict | None = None*, *mlp_net_kwargs: dict | None = None*, *use_avg_pooling: bool = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgCnnActor)

DDPG Convolutional Actor class.

Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
[https://arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf)

The DDPG Convolutional Actor takes as input an observation (some simple
transformation of the observed pixels) and returns an action vector from
it, as well as an observation embedding that can be reused for a value
estimation. It should be trained to maximise the value returned by the
DDPG Q Value network.

Parameters:

- **action_dim** (*int*) - length of the action vector.
- **conv_net_kwargs** (*dict**or**list**of**dicts**,**optional*) -

kwargs for the ConvNet.
Defaults to

```
>>> {
... 'in_features': None,
... "num_cells": [32, 64, 64],
... "kernel_sizes": [8, 4, 3],
... "strides": [4, 2, 1],
... "paddings": [0, 0, 1],
... 'activation_class': torch.nn.ELU,
... 'norm_class': None,
... 'aggregator_class': SquashDims,
... 'aggregator_kwargs': {"ndims_in": 3},
... 'squeeze_output': True,
... } #
```
- **mlp_net_kwargs** -

kwargs for MLP.
Defaults to:

```
>>> {
... 'in_features': None,
... 'out_features': action_dim,
... 'depth': 2,
... 'num_cells': 200,
... 'activation_class': nn.ELU,
... 'bias_last_layer': True,
... }
```
- **use_avg_pooling** (*bool**,**optional*) - if `True`, a
`AvgPooling` layer is used to aggregate the
output. Defaults to `False`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> import torch
>>> from torchrl.modules import DdpgCnnActor
>>> actor = DdpgCnnActor(action_dim=4)
>>> print(actor)
DdpgCnnActor(
 (convnet): ConvNet(
 (0): LazyConv2d(0, 32, kernel_size=(8, 8), stride=(4, 4))
 (1): ELU(alpha=1.0)
 (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
 (3): ELU(alpha=1.0)
 (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (5): ELU(alpha=1.0)
 (6): SquashDims()
 )
 (mlp): MLP(
 (0): LazyLinear(in_features=0, out_features=200, bias=True)
 (1): ELU(alpha=1.0)
 (2): Linear(in_features=200, out_features=200, bias=True)
 (3): ELU(alpha=1.0)
 (4): Linear(in_features=200, out_features=4, bias=True)
 )
)
>>> obs = torch.randn(10, 3, 64, 64)
>>> action, hidden = actor(obs)
>>> print(action.shape)
torch.Size([10, 4])
>>> print(hidden.shape)
torch.Size([10, 2304])
```

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/models.html#DdpgCnnActor.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.