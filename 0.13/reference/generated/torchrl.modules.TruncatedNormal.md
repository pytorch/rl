# TruncatedNormal

*class*torchrl.modules.TruncatedNormal(*loc: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *scale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *upscale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float = 5.0*, *low: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float = -1.0*, *high: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float = 1.0*, *tanh_loc: bool = False*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#TruncatedNormal)

Implements a Truncated Normal distribution with location scaling.

Location scaling prevents the location to be "too far" from 0, which ultimately
leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
In practice, the location is computed according to

> \[loc = tanh(loc / upscale) * upscale.\]

This behavior can be disabled by switching off the tanh_loc parameter (see below).

Parameters:

- **loc** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - normal distribution location parameter
- **scale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - normal distribution sigma parameter (squared root of variance)
- **upscale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) -

'a' scaling factor in the formula:

\[loc = tanh(loc / upscale) * upscale.\]

Default is 5.0
- **low** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) - minimum value of the distribution. Default = -1.0;
- **high** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) - maximum value of the distribution. Default = 1.0;
- **tanh_loc** (*bool**,**optional*) - if `True`, the above formula is used for
the location scaling, otherwise the raw value is kept.
Default is `False`;

log_prob(*value*, ***kwargs*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#TruncatedNormal.log_prob)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*mode

Returns the mode of the distribution.