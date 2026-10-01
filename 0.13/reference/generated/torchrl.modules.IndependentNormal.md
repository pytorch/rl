# IndependentNormal

*class*torchrl.modules.IndependentNormal(*loc: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *scale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float | Callable[[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*, *upscale: float = 5.0*, *tanh_loc: bool = False*, *event_dim: int = 1*, ***kwargs*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#IndependentNormal)

Implements a Normal distribution with location scaling.

Location scaling prevents the location to be "too far" from 0, which ultimately
leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
In practice, the location is computed according to

> \[loc = tanh(loc / upscale) * upscale.\]

This behavior can be disabled by switching off the tanh_loc parameter (see below).

Parameters:

- **loc** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - normal distribution location parameter
- **scale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**float**, or**callable*) - normal distribution sigma parameter (squared root of variance).
Can be a tensor, a float, or a callable that takes the `loc` tensor as input and returns the scale tensor.
Using a callable (e.g., `torch.ones_like` or `functools.partial(torch.full_like, fill_value=0.1)`)
avoids explicit device transfers like `torch.tensor(val, device=device)` and prevents graph breaks
in [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).
- **upscale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) -

'a' scaling factor in the formula:

\[loc = tanh(loc / upscale) * upscale.\]

Default is 5.0
- **tanh_loc** (*bool**,**optional*) - if `False`, the above formula is used for
the location scaling, otherwise the raw value
is kept. Default is `False`;

Example

```
>>> import torch
>>> from functools import partial
>>> from torchrl.modules.distributions import IndependentNormal
>>> loc = torch.zeros(3, 4)
>>> # Using a callable scale avoids device transfers and graph breaks in torch.compile
>>> dist = IndependentNormal(loc, scale=torch.ones_like)
>>> # For a custom scale value, use partial to create a callable
>>> dist = IndependentNormal(loc, scale=partial(torch.full_like, fill_value=0.1))
>>> sample = dist.sample()
>>> sample.shape
torch.Size([3, 4])
```

*property*mode

Returns the mode of the distribution.