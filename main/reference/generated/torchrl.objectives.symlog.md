# torchrl.objectives.symlog

torchrl.objectives.symlog(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/functional.html#symlog)

Apply the element-wise symmetric logarithm transform.

The transform is defined as
`sign(value) * log(1 + abs(value))` and compresses both positive and
negative values while remaining approximately linear around zero.

Parameters:

**value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Input tensor.

Returns:

A tensor with the same shape, dtype, and device as `value`.

Examples

```
>>> import torch
>>> from torchrl.modules import functional as F
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> F.symlog(value)
tensor([-4.6151, 0.0000, 4.6151])
```

See also

[`symexp()`](torchrl.objectives.symexp.html#torchrl.objectives.symexp) for the inverse operation and
[`SymLogValueTransform`](torchrl.modules.SymLogValueTransform.html#torchrl.modules.SymLogValueTransform) for the module form.