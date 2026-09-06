# SquashDims

*class*torchrl.modules.models.utils.SquashDims(*ndims_in: int = 3*)[[source]](../../_modules/torchrl/modules/models/utils.html#SquashDims)

A squashing layer.

Flattens the N last dimensions of an input tensor.

Parameters:

**ndims_in** (*int*) - number of dimensions to be flattened.
default = 3

Examples

```
>>> from torchrl.modules.models.utils import SquashDims
>>> import torch
>>> x = torch.randn(1, 2, 3, 4)
>>> print(SquashDims()(x).shape)
torch.Size([1, 24])
```

forward(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/utils.html#SquashDims.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.