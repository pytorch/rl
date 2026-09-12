# reset_noise

*class*torchrl.modules.reset_noise(*layer: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*)[[source]](../../_modules/torchrl/modules/models/exploration.html#reset_noise)

Resets the noise of noisy layers.

Designed to be passed to [`apply()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply):

```
module.apply(reset_noise)
```

[`NoisyLinear`](torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear) samples its factorized noise only
here and at construction. The forward pass does not resample.

Parameters:

**layer** (*nn.Module*) - a module. If it implements `reset_noise`, that
method is called; otherwise this is a no-op.