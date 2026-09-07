# DreamerV3ImageDecoder

*class*torchrl.modules.DreamerV3ImageDecoder(*in_features: int*, *image_shape: tuple[int, int, int] = (3, 64, 64)*, *depth: int = 64*, *mults: tuple[int, ...] = (2, 3, 4, 4)*, *kernel_size: int = 5*, *num_blocks: int = 8*, *norm_eps: float = 0.0001*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3ImageDecoder)

DreamerV3 transposed-convolution image decoder.

A (block-)linear projection maps the latent features to the smallest
feature map, then stride-2 transposed convolutions with channel-wise RMS
normalization and SiLU double the resolution at every stage. The last
layer outputs the image channels without normalization, shifted by
`0.5` to match the scale of image targets divided by `255`. Predictions
are unbounded.

Reference: Hafner et al., DreamerV3 (2023): [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **in_features** (*int*) - Latent feature count (for instance the stochastic
state concatenated with the belief).
- **image_shape** (*tuple**[**int**,**int**,**int**]**,**optional*) - Decoded `(C, H, W)`
shape. `H` and `W` must be divisible by `2 ** len(mults)`.
Defaults to `(3, 64, 64)`.
- **depth** (*int**,**optional*) - Base channel count, mirroring the encoder.
Defaults to 64.
- **mults** (*tuple**[**int**,**...**]**,**optional*) - Channel multipliers of the encoder
stages, mirrored here. Defaults to `(2, 3, 4, 4)`.
- **kernel_size** (*int**,**optional*) - Positive odd transposed convolution kernel
size. Defaults to 5.
- **num_blocks** (*int**,**optional*) - Feature blocks of the input projection
(see the block-linear layers of the reference implementation).
`1` uses a dense linear layer. Defaults to 8.
- **norm_eps** (*float**,**optional*) - RMS normalization epsilon. Defaults to
`1e-4`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device on which to create parameters.

Examples

```
>>> import torch
>>> from torchrl.modules import DreamerV3ImageDecoder
>>> decoder = DreamerV3ImageDecoder(
... in_features=12, image_shape=(3, 16, 16), depth=8, mults=(1, 2), num_blocks=2
... )
>>> decoder(torch.randn(4, 8), torch.randn(4, 4)).shape
torch.Size([4, 3, 16, 16])
```

See also

[`DreamerV3ImageDecoderConfig`](torchrl.trainers.algorithms.configs.modules.DreamerV3ImageDecoderConfig.html#torchrl.trainers.algorithms.configs.modules.DreamerV3ImageDecoderConfig)

forward(**inputs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3ImageDecoder.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.