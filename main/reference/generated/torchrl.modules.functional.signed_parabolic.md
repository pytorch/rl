# signed_parabolic

*class*torchrl.modules.functional.signed_parabolic(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *epsilon: float = 0.001*)[[source]](../../_modules/torchrl/modules/functional.html#signed_parabolic)

Apply the inverse of [`signed_hyperbolic()`](torchrl.modules.functional.signed_hyperbolic.html#torchrl.modules.functional.signed_hyperbolic) element-wise.

Parameters:

- **value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Input tensor in signed-hyperbolic space.
- **epsilon** (*float**,**optional*) - Positive linear correction used by the
corresponding [`signed_hyperbolic()`](torchrl.modules.functional.signed_hyperbolic.html#torchrl.modules.functional.signed_hyperbolic) call. Defaults to
`1e-3`.

Returns:

A tensor with the same shape, dtype, and device as `value`.

Examples

```
>>> import torch
>>> from torchrl.modules import functional as F
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> transformed = F.signed_hyperbolic(value)
>>> transformed
tensor([-9.1499, 0.0000, 9.1499])
>>> F.signed_parabolic(transformed)
tensor([-100.0000, 0.0000, 100.0000])
```

See also

[`signed_hyperbolic()`](torchrl.modules.functional.signed_hyperbolic.html#torchrl.modules.functional.signed_hyperbolic) for the forward operation and
[`SignedHyperbolicValueTransform`](torchrl.modules.SignedHyperbolicValueTransform.html#torchrl.modules.SignedHyperbolicValueTransform) for the module
form.