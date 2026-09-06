# TanhDelta

*class*torchrl.modules.TanhDelta(*param: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *low: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float = -1.0*, *high: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float = 1.0*, *event_dims: int = 1*, *atol: float = 1e-06*, *rtol: float = 1e-06*, *safe: bool = True*)[[source]](../../_modules/torchrl/modules/distributions/continuous.html#TanhDelta)

Implements a Tanh transformed_in Delta distribution.

Parameters:

- **param** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - parameter of the delta distribution;
- **low** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) - minimum value of the distribution. Default is -1.0;
- **high** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**number**,**optional*) - maximum value of the distribution. Default is 1.0;
- **event_dims** (*int**,**optional*) - number of dimensions describing the action.
Default is 1;
- **atol** (*number**,**optional*) - absolute tolerance to consider that a tensor matches the distribution parameter;
Default is 1e-6
- **rtol** (*number**,**optional*) - relative tolerance to consider that a tensor matches the distribution parameter;
Default is 1e-6
- **batch_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - batch shape;
- **event_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - shape of the outcome;

*property*mean*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mean of the distribution.

*property*mode*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mode of the distribution.