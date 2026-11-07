# signed_hyperbolic

*class*torchrl.modules.functional.signed_hyperbolic(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *epsilon: float = 0.001*)[[source]](../../_modules/torchrl/modules/functional.html#signed_hyperbolic)

Apply the signed hyperbolic value transform.

This is the scale-compressing transform introduced by Pohlen et al. and
used by algorithms in the MuZero and Muesli families:

`sign(value) * (sqrt(abs(value) + 1) - 1) + epsilon * value`.

Parameters:

- **value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Input tensor.
- **epsilon** (*float**,**optional*) - Positive linear correction that keeps the
inverse Lipschitz continuous. Defaults to `1e-3`.

Returns:

A tensor with the same shape, dtype, and device as `value`.

Examples

```
>>> import torch
>>> from torchrl.modules import functional as F
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> F.signed_hyperbolic(value)
tensor([-9.1499, 0.0000, 9.1499])
```

See also

[`signed_parabolic()`](torchrl.modules.functional.signed_parabolic.html#torchrl.modules.functional.signed_parabolic) for the inverse operation and
[`SignedHyperbolicValueTransform`](torchrl.modules.SignedHyperbolicValueTransform.html#torchrl.modules.SignedHyperbolicValueTransform) for the module
form.

Note

See [Observe and Look Further: Achieving Consistent Performance on
Atari](https://arxiv.org/abs/1805.11593) (Pohlen et al., 2018).