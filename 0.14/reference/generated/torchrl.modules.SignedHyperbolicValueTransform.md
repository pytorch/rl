# SignedHyperbolicValueTransform

*class*torchrl.modules.SignedHyperbolicValueTransform(*epsilon: float = 0.001*)[[source]](../../_modules/torchrl/modules/value_transforms.html#SignedHyperbolicValueTransform)

Signed-hyperbolic transform for large-magnitude value targets.

Parameters:

**epsilon** (*float**,**optional*) - Positive linear correction that keeps the
inverse Lipschitz continuous. Defaults to `1e-3`.

Examples

```
>>> import torch
>>> from torchrl.modules import SignedHyperbolicValueTransform
>>> transform = SignedHyperbolicValueTransform(epsilon=1e-3)
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> transformed = transform(value)
>>> transformed
tensor([-9.1499, 0.0000, 9.1499])
>>> transform.inverse(transformed)
tensor([-100.0000, 0.0000, 100.0000])
```

See also

[`torchrl.modules.functional.signed_hyperbolic()`](torchrl.modules.functional.signed_hyperbolic.html#torchrl.modules.functional.signed_hyperbolic) and
[`torchrl.modules.functional.signed_parabolic()`](torchrl.modules.functional.signed_parabolic.html#torchrl.modules.functional.signed_parabolic) for the functional
form, and [`SymLogValueTransform`](torchrl.modules.SymLogValueTransform.html#torchrl.modules.SymLogValueTransform) for an alternative nonlinear
transform.

Note

See [Observe and Look Further: Achieving Consistent Performance on
Atari](https://arxiv.org/abs/1805.11593) (Pohlen et al., 2018).

forward(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#SignedHyperbolicValueTransform.forward)

Apply the signed-hyperbolic transform to `value`.

inverse(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#SignedHyperbolicValueTransform.inverse)

Apply the signed-parabolic inverse to `value`.