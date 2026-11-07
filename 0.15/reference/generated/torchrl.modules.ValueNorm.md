# ValueNorm

*class*torchrl.modules.ValueNorm(***, *shape: int | tuple[int, ...] = 1*, *epsilon: float = 1e-05*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*)[[source]](../../_modules/torchrl/modules/value_norm.html#ValueNorm)

Abstract base class for value normalisers.

A *value normaliser* keeps a running estimate of the location and scale of
the value target seen during training. Critics use it to:

- **normalize** the regression target before computing MSE, keeping the
critic loss on a fixed scale across episodes / reward inflations;
- **denormalize** the critic's output back to the real reward scale when
forming bootstrapped value estimates inside GAE / TD.

Subclasses must implement `update()`, `normalize()`,
`denormalize()`, and `scale()`. The convention is that they all
operate on tensors whose trailing dims match `shape` (the
per-element value shape, usually `(1,)`).

*abstract*denormalize(*normalised_value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#ValueNorm.denormalize)

Inverse of `normalize()` -- recover real-scale values.

*abstract*normalize(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#ValueNorm.normalize)

Standardise `value_target` using the current running stats.

scale() → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#ValueNorm.scale)

Multiplicative scale currently applied by `normalize()`.

Exposed separately so consumers can rescale quantities that must not
be re-centred, e.g. advantages (already centred by the value
baseline), for which only the division by the scale applies.

Deliberately not abstract until v0.16 so that subclasses written
before it existed keep instantiating (with a `DeprecationWarning`);
this default raises `NotImplementedError` when called.

*abstract*update(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/modules/value_norm.html#ValueNorm.update)

Fold a batch of value targets into the running stats.