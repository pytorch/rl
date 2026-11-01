# NormalParamExtractor

*class*torchrl.modules.NormalParamExtractor(*scale_mapping: str = 'biased_softplus_1.0'*, *scale_lb: Number = 0.0001*)[[source]](../../_modules/tensordict/nn/distributions/continuous.html#NormalParamExtractor)

A module without parameters that splits a tensor into the location and scale of a normal distribution.

The module splits the last dimension of its first input into two halves:
the first half is the location, the second half goes through
`scale_mapping` and is clamped to at least `scale_lb` to give the
scale. With an odd size, the location gets one element more than the
scale. Calling the module with
more inputs returns them unchanged after the scale:
`module(x, *others)` returns `(loc, scale, *others)`.

Parameters:

- **scale_mapping** (*str**,**optional*) - the function that maps the second half
to positive values, by name: for example `"softplus"`,
`"exp"`, `"relu"`, `"biased_softplus_<bias>"` or `"none"`
(no mapping). See [`mappings()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.mappings.html#tensordict.nn.mappings). Defaults to
`"biased_softplus_1.0"`, a softplus shifted so that an input of
`0` gives a scale of `1`.
- **scale_lb** (*Number**,**optional*) - the minimum value of the scale (the
standard deviation). Defaults to `1e-4`.

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