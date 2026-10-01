# ObsEncoder

*class*torchrl.modules.ObsEncoder(*channels=32*, *num_layers=4*, *in_channels=None*, *depth=None*, *device=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#ObsEncoder)

Observation encoder network.

Takes a pixel observation and encodes it into a latent space.

Reference: [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

Parameters:

- **channels** (*int**,**optional*) - Number of hidden units in the first layer.
Defaults to 32.
- **num_layers** (*int**,**optional*) - Depth of the network. Defaults to 4.
- **in_channels** (*int**,**optional*) - Number of input channels. If None, uses LazyConv2d.
Defaults to None for backward compatibility.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device to create the module on.
Defaults to None (uses default device).

forward(*observation*)[[source]](../../_modules/torchrl/modules/models/model_based.html#ObsEncoder.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.