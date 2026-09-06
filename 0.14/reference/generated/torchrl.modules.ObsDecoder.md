# ObsDecoder

*class*torchrl.modules.ObsDecoder(*channels=32*, *num_layers=4*, *kernel_sizes=None*, *latent_dim=None*, *out_channels=3*, *depth=None*, *device=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#ObsDecoder)

Observation decoder network.

Takes the deterministic state and the stochastic belief and decodes it into a pixel observation.

Reference: [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

Parameters:

- **channels** (*int**,**optional*) - Number of hidden units in the last layer.
Defaults to 32.
- **num_layers** (*int**,**optional*) - Depth of the network. Defaults to 4.
- **kernel_sizes** (*int**or**list**of**int**,**optional*) - the kernel_size of each layer.
Defaults to `[5, 5, 6, 6]` if num_layers if 4, else `[5] * num_layers`.
- **latent_dim** (*int**,**optional*) - Input dimension (state_dim + rnn_hidden_dim).
If None, uses LazyLinear. Defaults to None for backward compatibility.
- **out_channels** (*int**,**optional*) - Number of output channels in the final
ConvTranspose2d layer. Defaults to 3 (RGB). Set to 1 for
grayscale.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device to create the module on.
Defaults to None (uses default device).

forward(*state*, *rnn_hidden*)[[source]](../../_modules/torchrl/modules/models/model_based.html#ObsDecoder.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.