# ComposeValueTransform

*class*torchrl.modules.ComposeValueTransform(**transforms: [ValueTransform](torchrl.modules.ValueTransform.html#torchrl.modules.ValueTransform)*)[[source]](../../_modules/torchrl/modules/value_transforms.html#ComposeValueTransform)

Compose value transforms while preserving the inverse mapping.

Forward transforms are applied in the order provided. Inverse transforms
are applied in reverse order.

Parameters:

***transforms** ([*ValueTransform*](torchrl.modules.ValueTransform.html#torchrl.modules.ValueTransform)) - Transforms to compose.

Examples

```
>>> import torch
>>> from torchrl.modules import (
... ComposeValueTransform,
... SignedHyperbolicValueTransform,
... SymLogValueTransform,
... )
>>> transform = ComposeValueTransform(
... SignedHyperbolicValueTransform(), SymLogValueTransform()
... )
>>> value = torch.tensor([-100.0, 0.0, 100.0])
>>> transformed = transform(value)
>>> transformed
tensor([-2.3175, 0.0000, 2.3175])
>>> transform.inverse(transformed)
tensor([-100.0000, 0.0000, 100.0000])
```

See also

[`ValueTransform`](torchrl.modules.ValueTransform.html#torchrl.modules.ValueTransform) for the component interface,
[`SymLogValueTransform`](torchrl.modules.SymLogValueTransform.html#torchrl.modules.SymLogValueTransform), and
[`SignedHyperbolicValueTransform`](torchrl.modules.SignedHyperbolicValueTransform.html#torchrl.modules.SignedHyperbolicValueTransform).

forward(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#ComposeValueTransform.forward)

Apply the component transforms in order.

inverse(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#ComposeValueTransform.inverse)

Apply the component inverse transforms in reverse order.