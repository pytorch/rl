# ValueTransform

*class*torchrl.modules.ValueTransform(**args: Any*, ***kwargs: Any*)[[source]](../../_modules/torchrl/modules/value_transforms.html#ValueTransform)

Abstract base class for invertible scalar value transforms.

A value transform maps raw rewards or returns to a numerically convenient
prediction space. `inverse()` maps predictions back to the raw value
space before they are used for bootstrapping.

Subclasses implement `forward()` and `inverse()` as element-wise
tensor operations.

Examples

```
>>> import torch
>>> from torchrl.modules import ValueTransform
>>> class ScaleValueTransform(ValueTransform):
... def forward(self, value):
... return value * 2
... def inverse(self, value):
... return value / 2
>>> transform = ScaleValueTransform()
>>> value = torch.tensor([-2.0, 0.0, 2.0])
>>> transformed = transform(value)
>>> transformed
tensor([-4., 0., 4.])
>>> transform.inverse(transformed)
tensor([-2., 0., 2.])
```

See also

[`IdentityValueTransform`](torchrl.modules.IdentityValueTransform.html#torchrl.modules.IdentityValueTransform), [`SymLogValueTransform`](torchrl.modules.SymLogValueTransform.html#torchrl.modules.SymLogValueTransform),
[`SignedHyperbolicValueTransform`](torchrl.modules.SignedHyperbolicValueTransform.html#torchrl.modules.SignedHyperbolicValueTransform), and
[`ComposeValueTransform`](torchrl.modules.ComposeValueTransform.html#torchrl.modules.ComposeValueTransform).

*abstract*forward(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#ValueTransform.forward)

Map a raw value tensor to the transformed prediction space.

*abstract*inverse(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#ValueTransform.inverse)

Map a transformed value tensor back to raw value space.