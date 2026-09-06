# DdpgMlpQNet

*class*torchrl.modules.DdpgMlpQNet(*mlp_net_kwargs_net1: dict | None = None*, *mlp_net_kwargs_net2: dict | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgMlpQNet)

DDPG Q-value MLP class.

Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
[https://arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf)

The DDPG Q-value network takes as input an observation and an action,
and returns a scalar from it.
Because actions are integrated later than observations, two networks are
created.

Parameters:

- **mlp_net_kwargs_net1** (*dict**,**optional*) -

kwargs for MLP.
Defaults to

```
>>> {
... 'in_features': None,
... 'out_features': 400,
... 'depth': 0,
... 'num_cells': [],
... 'activation_class': nn.ELU,
... 'bias_last_layer': True,
... 'activate_last_layer': True,
... }
```
- **mlp_net_kwargs_net2** -

Defaults to

```
>>> {
... 'in_features': None,
... 'out_features': 1,
... 'depth': 1,
... 'num_cells': [300, ],
... 'activation_class': nn.ELU,
... 'bias_last_layer': True,
... }
```
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> import torch
>>> from torchrl.modules import DdpgMlpQNet
>>> net = DdpgMlpQNet()
>>> print(net)
DdpgMlpQNet(
 (mlp1): MLP(
 (0): LazyLinear(in_features=0, out_features=400, bias=True)
 (1): ELU(alpha=1.0)
 )
 (mlp2): MLP(
 (0): LazyLinear(in_features=0, out_features=300, bias=True)
 (1): ELU(alpha=1.0)
 (2): Linear(in_features=300, out_features=1, bias=True)
 )
)
>>> obs = torch.zeros(1, 32)
>>> action = torch.zeros(1, 4)
>>> value = net(obs, action)
>>> print(value.shape)
torch.Size([1, 1])
```

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgMlpQNet.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.