# Delta

*class*torchrl.modules.Delta(*param: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *atol: float = 1e-06*, *rtol: float = 1e-06*, *batch_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | Sequence[int] = None*, *event_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | Sequence[int] = None*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#Delta)

Delta distribution.

Parameters:

- **param** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - parameter of the delta distribution;
- **atol** (*number**,**optional*) - absolute tolerance to consider that a tensor matches the distribution parameter;
Default is 1e-6
- **rtol** (*number**,**optional*) - relative tolerance to consider that a tensor matches the distribution parameter;
Default is 1e-6
- **batch_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - batch shape;
- **event_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - shape of the outcome.

expand(*batch_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*, *_instance=None*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#Delta.expand)

Returns a new distribution instance (or populates an existing instance
provided by a derived class) with batch dimensions expanded to
batch_shape. This method calls `expand` on
the distribution's parameters. As such, this does not allocate new
memory for the expanded distribution instance. Additionally,
this does not repeat any args checking or parameter broadcasting in
__init__.py, when an instance is first created.

Parameters:

- **batch_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the desired expanded size.
- **_instance** - new instance provided by subclasses that
need to override .expand.

Returns:

New distribution instance with batch dimensions expanded to
batch_size.

log_prob(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#Delta.log_prob)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*mean*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mean of the distribution.

*property*mode*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mode of the distribution.

rsample(*size=None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#Delta.rsample)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

sample(*size=None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#Delta.sample)

Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.