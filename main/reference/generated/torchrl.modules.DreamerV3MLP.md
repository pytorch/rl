# DreamerV3MLP

*class*torchrl.modules.DreamerV3MLP(*in_features: int*, *out_features: int*, *depth: int = 3*, *num_cells: int = 1024*, *outscale: float = 1.0*, *norm_eps: float = 0.0001*, *device=None*)[[source]](../../_modules/torchrl/modules/models/model_based_v3.html#DreamerV3MLP)

RMS-normalized multilayer perceptron used by DreamerV3 heads.

Parameters:

- **in_features** (*int*) - Input feature count.
- **out_features** (*int*) - Output feature count.
- **depth** (*int**,**optional*) - Number of hidden layers. Defaults to 3.
- **num_cells** (*int**,**optional*) - Hidden feature count. Defaults to 1024.
- **outscale** (*float**,**optional*) - Multiplicative initialization scale for the
output layer. Defaults to 1.0.
- **norm_eps** (*float**,**optional*) - RMS normalization epsilon. Defaults to
`1e-4`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device on which to create parameters.

Examples

```
>>> import torch
>>> from torchrl.modules import DreamerV3MLP
>>> module = DreamerV3MLP(6, 4, depth=2, num_cells=8)
>>> module(torch.randn(3, 2), torch.randn(3, 4)).shape
torch.Size([3, 4])
```

forward(**inputs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based_v3.html#DreamerV3MLP.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.