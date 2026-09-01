# DuelingCnnDQNet

*class*torchrl.modules.DuelingCnnDQNet(*out_features: int*, *out_features_value: int = 1*, *cnn_kwargs: dict | None = None*, *mlp_kwargs: dict | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#DuelingCnnDQNet)

Dueling CNN Q-network.

Presented in [https://arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581)

Parameters:

- **out_features** (*int*) - number of features for the advantage network.
- **out_features_value** (*int*) - number of features for the value network.
- **cnn_kwargs** (*dict**or**list**of**dicts**,**optional*) -

kwargs for the feature
network. Default is

```
>>> cnn_kwargs = {
... 'num_cells': [32, 64, 64],
... 'strides': [4, 2, 1],
... 'kernel_sizes': [8, 4, 3],
... }
```
- **mlp_kwargs** (*dict**or**list**of**dicts**,**optional*) -

kwargs for the advantage
and value network. Default is

```
>>> mlp_kwargs = {
... "depth": 1,
... "activation_class": nn.ELU,
... "num_cells": 512,
... "bias_last_layer": True,
... }
```
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> import torch
>>> from torchrl.modules import DuelingCnnDQNet
>>> net = DuelingCnnDQNet(out_features=20)
>>> print(net)
DuelingCnnDQNet(
 (features): ConvNet(
 (0): LazyConv2d(0, 32, kernel_size=(8, 8), stride=(4, 4))
 (1): ELU(alpha=1.0)
 (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
 (3): ELU(alpha=1.0)
 (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
 (5): ELU(alpha=1.0)
 (6): SquashDims()
 )
 (advantage): MLP(
 (0): LazyLinear(in_features=0, out_features=512, bias=True)
 (1): ELU(alpha=1.0)
 (2): Linear(in_features=512, out_features=20, bias=True)
 )
 (value): MLP(
 (0): LazyLinear(in_features=0, out_features=512, bias=True)
 (1): ELU(alpha=1.0)
 (2): Linear(in_features=512, out_features=1, bias=True)
 )
)
>>> x = torch.zeros(1, 3, 64, 64)
>>> y = net(x)
>>> print(y.shape)
torch.Size([1, 20])
```

forward(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#DuelingCnnDQNet.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.