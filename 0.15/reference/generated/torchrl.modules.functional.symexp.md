# symexp

*class*torchrl.modules.functional.symexp(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../../_modules/torchrl/modules/functional.html#symexp)

Apply the inverse symmetric exponential transform element-wise.

Parameters:

**value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Input tensor in symmetric-log space.

Returns:

A tensor with the same shape, dtype, and device as `value`.

Examples

```
>>> import torch
>>> from torchrl.modules import functional as F
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> transformed = F.symlog(value)
>>> transformed
tensor([-4.6151, 0.0000, 4.6151])
>>> F.symexp(transformed)
tensor([-100.0000, 0.0000, 100.0000])
```

See also

[`symlog()`](torchrl.modules.functional.symlog.html#torchrl.modules.functional.symlog) for the forward operation and
[`SymLogValueTransform`](torchrl.modules.SymLogValueTransform.html#torchrl.modules.SymLogValueTransform) for the module form.