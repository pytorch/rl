# TanhNormal

*class*torchrl.modules.TanhNormal(*loc: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *scale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float | Callable[[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*, *upscale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | Number = 5.0*, *low: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | Number = -1.0*, *high: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | Number = 1.0*, *event_dims: int | None = None*, *tanh_loc: bool = False*, *safe_tanh: bool = True*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#TanhNormal)

Implements a TanhNormal distribution with location scaling.

Location scaling prevents the location to be "too far" from 0 when a
`TanhTransform` is applied, but ultimately
leads to numerically unstable samples and poor gradient computation
(e.g. gradient explosion).
In practice, with location scaling the location is computed according to

> \[loc = tanh(loc / upscale) * upscale.\]

Parameters:

- **loc** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - normal distribution location parameter
- **scale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**float**, or**callable*) - normal distribution sigma parameter (squared root of variance).
Can be a tensor, a float, or a callable that takes the `loc` tensor as input and returns the scale tensor.
Using a callable (e.g., `torch.ones_like` or `functools.partial(torch.full_like, fill_value=0.1)`)
avoids explicit device transfers like `torch.tensor(val, device=device)` and prevents graph breaks
in [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).
- **upscale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number*) -

'a' scaling factor in the formula:

\[loc = tanh(loc / upscale) * upscale.\]
- **low** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) - minimum value of the distribution. Default is -1.0;
- **high** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) - maximum value of the distribution. Default is 1.0;
- **event_dims** (*int**,**optional*) - number of dimensions describing the action.
Default is 1. Setting `event_dims` to `0` will result in a log-probability that has the same shape
as the input, `1` will reduce (sum over) the last dimension, `2` the last two etc.
- **tanh_loc** (*bool**,**optional*) - if `True`, the above formula is used for the location scaling, otherwise the raw
value is kept. Default is `False`;
- **safe_tanh** (*bool**,**optional*) - if `True`, the Tanh transform is done "safely", to avoid numerical overflows.
This will currently break with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

Example

```
>>> import torch
>>> from functools import partial
>>> from torchrl.modules.distributions import TanhNormal
>>> loc = torch.zeros(3, 4)
>>> # Using a callable scale avoids device transfers and graph breaks in torch.compile
>>> dist = TanhNormal(loc, scale=torch.ones_like)
>>> # For a custom scale value, use partial to create a callable
>>> dist = TanhNormal(loc, scale=partial(torch.full_like, fill_value=0.1))
>>> sample = dist.sample()
>>> sample.shape
torch.Size([3, 4])
```

get_mode()[[source]](../../_modules/torchrl/modules/distributions/continuous.html#TanhNormal.get_mode)

Computes an estimation of the mode using the Adam optimizer.

*property*mean

Returns the mean of the distribution.

*property*mode

Returns the mode of the distribution.

*property*support

Returns a [`Constraint`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint) object
representing this distribution's support.