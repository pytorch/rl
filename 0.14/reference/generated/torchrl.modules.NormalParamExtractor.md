# NormalParamExtractor

*class*torchrl.modules.NormalParamExtractor(*scale_mapping: str = 'biased_softplus_1.0'*, *scale_lb: Number = 0.0001*)[[source]](../../_modules/tensordict/nn/distributions/continuous.html#NormalParamExtractor)

A non-parametric nn.Module that splits its input into loc and scale parameters.

The scale parameters are mapped onto positive values using the specified `scale_mapping`.

Parameters:

- **scale_mapping** (*str**,**optional*) - positive mapping function to be used with the std.
default = `"biased_softplus_1.0"` (i.e. softplus map with bias such that fn(0.0) = 1.0)
choices: `"softplus"`, `"exp"`, `"relu"`, `"biased_softplus_1"` or `"none"` (no mapping).
See [`mappings()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.mappings.html#tensordict.nn.mappings) for more details.
- **scale_lb** (*Number**,**optional*) - The minimum value that the variance can take. Default is 1e-4.

Examples

```
>>> import torch
>>> from tensordict.nn.distributions import NormalParamExtractor
>>> from torch import nn
>>> module = nn.Linear(3, 4)
>>> normal_params = NormalParamExtractor()
>>> tensor = torch.randn(3)
>>> loc, scale = normal_params(module(tensor))
>>> print(loc.shape, scale.shape)
torch.Size([2]) torch.Size([2])
>>> assert (scale > 0).all()
>>> # with modules that return more than one tensor
>>> module = nn.LSTM(3, 4)
>>> tensor = torch.randn(4, 2, 3)
>>> loc, scale, others = normal_params(*module(tensor))
>>> print(loc.shape, scale.shape)
torch.Size([4, 2, 2]) torch.Size([4, 2, 2])
>>> assert (scale > 0).all()
```

forward(**tensors: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...][[source]](../../_modules/tensordict/nn/distributions/continuous.html#NormalParamExtractor.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.