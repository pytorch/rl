# SymExpTwoHot

*class*torchrl.modules.SymExpTwoHot(*num_bins: int = 255*)[[source]](../../_modules/torchrl/modules/models/dreamer_v3.html#SymExpTwoHot)

DreamerV3 categorical scalar representation.

The support contains `num_bins` raw scalar values obtained by applying
`symexp` to an evenly spaced grid from -20 to 20. Targets are interpolated
between adjacent raw support values, while predictions are decoded as the
softmax-weighted raw-value expectation.

Parameters:

**num_bins** (*int**,**optional*) - Number of categorical support values.
Defaults to 255.

Examples

```
>>> import torch
>>> from torchrl.modules import SymExpTwoHot
>>> two_hot = SymExpTwoHot(num_bins=5)
>>> target = torch.tensor([-10.0, 0.0, 10.0])
>>> encoded = two_hot.encode(target)
>>> decoded = two_hot.decode(encoded.log())
>>> torch.allclose(decoded, target, atol=1e-3)
True
```

decode(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/dreamer_v3.html#SymExpTwoHot.decode)

Decode categorical logits to raw scalar values.

encode(*target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/dreamer_v3.html#SymExpTwoHot.encode)

Encode raw scalar targets as two-hot categorical targets.

forward(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/dreamer_v3.html#SymExpTwoHot.forward)

Decode logits and retain a trailing scalar event dimension.

loss(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/dreamer_v3.html#SymExpTwoHot.loss)

Compute two-hot cross entropy against raw scalar targets.