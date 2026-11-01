# SymLogValueTransform

*class*torchrl.modules.SymLogValueTransform(**args: Any*, ***kwargs: Any*)[[source]](../../_modules/torchrl/modules/value_transforms.html#SymLogValueTransform)

Symmetric-log value transform used by DreamerV3.

This transform applies [`torchrl.modules.functional.symlog()`](torchrl.modules.functional.symlog.html#torchrl.modules.functional.symlog) in the
forward direction and [`torchrl.modules.functional.symexp()`](torchrl.modules.functional.symexp.html#torchrl.modules.functional.symexp) in the
inverse direction.

Examples

```
>>> import torch
>>> from torchrl.modules import SymLogValueTransform
>>> transform = SymLogValueTransform()
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> transformed = transform(value)
>>> transformed
tensor([-4.6151, 0.0000, 4.6151])
>>> transform.inverse(transformed)
tensor([-100.0000, 0.0000, 100.0000])
```

See also

[`torchrl.modules.functional.symlog()`](torchrl.modules.functional.symlog.html#torchrl.modules.functional.symlog) and
[`torchrl.modules.functional.symexp()`](torchrl.modules.functional.symexp.html#torchrl.modules.functional.symexp) for the functional form,
and [`SignedHyperbolicValueTransform`](torchrl.modules.SignedHyperbolicValueTransform.html#torchrl.modules.SignedHyperbolicValueTransform) for an alternative
nonlinear transform.

Note

See [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023).

forward(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#SymLogValueTransform.forward)

Apply [`torchrl.modules.functional.symlog()`](torchrl.modules.functional.symlog.html#torchrl.modules.functional.symlog) to `value`.

inverse(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#SymLogValueTransform.inverse)

Apply [`torchrl.modules.functional.symexp()`](torchrl.modules.functional.symexp.html#torchrl.modules.functional.symexp) to `value`.