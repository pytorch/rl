# NoisyLinear

*class*torchrl.modules.NoisyLinear(*in_features: int*, *out_features: int*, *bias: bool = True*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *std_init: float = 0.5*, *use_exploration_type: bool | None = True*)[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLinear)

Noisy Linear Layer.

Presented in "Noisy Networks for Exploration" (Fortunato et al., 2017),
[https://arxiv.org/abs/1706.10295v3](https://arxiv.org/abs/1706.10295v3)

A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
with gradient descent along with any other remaining network weights. Factorized Gaussian
noise is the type of noise usually employed.

Note

The noise is controlled by the exploration mode set via
`set_exploration_type()`. When exploration type is
`RANDOM`, noise is added to the weights.
When exploration type is `DETERMINISTIC`,
`MODE`, or
`MEAN`, only the mean weights are used.

This behavior is controlled by the `use_exploration_type` argument. When set to
`True`, the exploration type is used. When set to `False`, the legacy behavior
of using `self.training` (i.e., `model.train()`/`model.eval()`) is used instead.

Note

Factorized noise is sampled only in
`reset_noise()`. The
forward pass does not resample. The same `weight_epsilon` /
`bias_epsilon` buffers are reused until the caller resamples them
with `layer.reset_noise()` or `module.apply(reset_noise)`.

The paper samples a new set of parameters after every optimization
step (section 3.1). Callers that want that cadence should apply
[`reset_noise()`](torchrl.modules.reset_noise.html#torchrl.modules.reset_noise) after each optimizer step:

```
module.apply(reset_noise)
```

[`make_trainer()`](torchrl.trainers.helpers.make_trainer.html#torchrl.trainers.helpers.make_trainer) uses a coarser
cadence: when `cfg.noisy` is set it registers
`loss_module.apply(reset_noise)` on the trainer's
`pre_optim_steps` hook, which runs once per
`optim_steps()` call, not after
every inner optimizer step, and only on `loss_module`.
Do not resample on every forward: that would change the behavior of
every NoisyNet user, including data collection, where a fixed sample
of the noisy weights is intended.

Parameters:

- **in_features** (*int*) - input features dimension
- **out_features** (*int*) - out features dimension
- **bias** (*bool**,**optional*) - if `True`, a bias term will be added to the matrix multiplication: Ax + b.
Defaults to `True`
- **device** (*DEVICE_TYPING**,**optional*) - device of the layer.
Defaults to `"cpu"`
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype of the parameters.
Defaults to `None` (default pytorch dtype)
- **std_init** (*scalar**,**optional*) - initial value of the Gaussian standard deviation before optimization.
Defaults to `0.5` as per the original paper.
- **use_exploration_type** (*bool**or**None**,**optional*) - if `True`, noise is controlled by
`exploration_type()`. If `False`, noise is controlled
by `self.training` (legacy behavior). If `None`, it is treated as `True`.
Defaults to `True`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.envs import ExplorationType, set_exploration_type
>>> from torchrl.modules import NoisyLinear, reset_noise
>>> _ = torch.manual_seed(0)
>>> layer = NoisyLinear(4, 2)
>>> x = torch.ones(4)
>>> with set_exploration_type(ExplorationType.RANDOM):
... y0 = layer(x)
... y1 = layer(x)
>>> torch.equal(y0, y1)
True
>>> with set_exploration_type(ExplorationType.RANDOM):
... layer.reset_noise()
... y2 = layer(x)
>>> torch.equal(y0, y2)
False
>>> net = nn.Sequential(NoisyLinear(4, 8), nn.ReLU(), NoisyLinear(8, 2))
>>> _ = net.apply(reset_noise)
```

reset_noise() → None[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLinear.reset_noise)

Resample the factorized Gaussian noise buffers.

Called from `__init__()`. The forward
pass does not call this method; apply it after each optimization step
(for example `module.apply(reset_noise)`).

reset_parameters() → None[[source]](../../_modules/torchrl/modules/models/exploration.html#NoisyLinear.reset_parameters)

Resets parameters based on their initialization used in `__init__`.