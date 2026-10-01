# DdpgMlpActor

*class*torchrl.modules.DdpgMlpActor(*action_dim: int*, *mlp_net_kwargs: dict | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgMlpActor)

DDPG Actor class.

Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
[https://arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf)

The DDPG Actor takes as input an observation vector and returns an action from it.
It is trained to maximise the value returned by the DDPG Q Value network.

Parameters:

- **action_dim** (*int*) - length of the action vector
- **mlp_net_kwargs** (*dict**,**optional*) -

kwargs for MLP.
Defaults to

```
>>> {
... 'in_features': None,
... 'out_features': action_dim,
... 'depth': 2,
... 'num_cells': [400, 300],
... 'activation_class': nn.ELU,
... 'bias_last_layer': True,
... }
```
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> import torch
>>> from torchrl.modules import DdpgMlpActor
>>> actor = DdpgMlpActor(action_dim=4)
>>> print(actor)
DdpgMlpActor(
 (mlp): MLP(
 (0): LazyLinear(in_features=0, out_features=400, bias=True)
 (1): ELU(alpha=1.0)
 (2): Linear(in_features=400, out_features=300, bias=True)
 (3): ELU(alpha=1.0)
 (4): Linear(in_features=300, out_features=4, bias=True)
 )
)
>>> obs = torch.zeros(10, 6)
>>> action = actor(obs)
>>> print(action.shape)
torch.Size([10, 4])
```

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#DdpgMlpActor.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.