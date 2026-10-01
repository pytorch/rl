# NoisyLazyLinear

*class*torchrl.modules.NoisyLazyLinear(*out_features: int*, *bias: bool = True*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *std_init: float = 0.5*, *use_exploration_type: bool | None = True*)[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLazyLinear)

Noisy Lazy Linear Layer.

This class makes the Noisy Linear layer lazy, in that the in_feature argument does not need to be passed at
initialization (but is inferred after the first call to the layer).

For more context on noisy layers, see the NoisyLinear class.
Like [`NoisyLinear`](torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear), noise is sampled only by
[`reset_noise()`](torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear.reset_noise) (at materialization)
and must be reapplied by the caller.

Parameters:

- **out_features** (*int*) - out features dimension
- **bias** (*bool**,**optional*) - if `True`, a bias term will be added to the matrix multiplication: Ax + b.
Defaults to `True`.
- **device** (*DEVICE_TYPING**,**optional*) - device of the layer.
Defaults to `"cpu"`.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype of the parameters.
Defaults to the default PyTorch dtype.
- **std_init** (*scalar*) - initial value of the Gaussian standard deviation before optimization.
Defaults to `0.5` as per the original paper.
- **use_exploration_type** (*bool**or**None**,**optional*) - if `True`, noise is controlled by
`exploration_type()`. If `False`, noise is controlled
by `self.training` (legacy behavior). If `None`, it is treated as `True`.
Defaults to `True`.

cls_to_become

alias of [`NoisyLinear`](torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear)

initialize_parameters(*input: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLazyLinear.initialize_parameters)

Initialize parameters according to the input batch properties.

This adds an interface to isolate parameter initialization from the
forward pass when doing parameter shape inference.

reset_noise() → None[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLazyLinear.reset_noise)

Resample the factorized Gaussian noise buffers.

Called from `__init__()`. The forward
pass does not call this method; apply it after each optimization step
(for example `module.apply(reset_noise)`).

reset_parameters() → None[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLazyLinear.reset_parameters)

Resets parameters based on their initialization used in `__init__`.