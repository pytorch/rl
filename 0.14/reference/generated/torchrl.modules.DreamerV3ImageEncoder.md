# DreamerV3ImageEncoder

*class*torchrl.modules.DreamerV3ImageEncoder(*in_channels: int = 3*, *depth: int = 64*, *mults: tuple[int, ...] = (2, 3, 4, 4)*, *kernel_size: int = 5*, *norm_eps: float = 0.0001*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3ImageEncoder)

DreamerV3 convolutional image encoder.

A stack of stride-2 convolutions, each followed by channel-wise RMS
normalization and SiLU, as in the reference implementation. Every stage
halves the spatial resolution and outputs `depth * mult` channels.

Reference: Hafner et al., DreamerV3 (2023): [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **in_channels** (*int**,**optional*) - Image channels. Defaults to 3.
- **depth** (*int**,**optional*) - Base channel count; stage `i` outputs
`depth * mults[i]` channels. Defaults to 64.
- **mults** (*tuple**[**int**,**...**]**,**optional*) - Channel multiplier of each stage.
Defaults to `(2, 3, 4, 4)`.
- **kernel_size** (*int**,**optional*) - Positive odd convolution kernel size. Defaults to 5.
- **norm_eps** (*float**,**optional*) - RMS normalization epsilon. Defaults to
`1e-4`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device on which to create parameters.

The input is an image batch of shape `(*batch, C, H, W)`, either
`uint8` in `[0, 255]` or floating point in `[0, 1]`. Both are mapped
to `[-0.5, 0.5]` before the first convolution. The output is the
flattened final feature map, `(*batch, output_features((C, H, W)))`.

Examples

```
>>> import torch
>>> from torchrl.modules import DreamerV3ImageEncoder
>>> encoder = DreamerV3ImageEncoder(depth=8, mults=(1, 2))
>>> image = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
>>> encoder(image).shape
torch.Size([4, 256])
>>> encoder.output_features((3, 16, 16))
256
```

See also

[`DreamerV3ImageEncoderConfig`](torchrl.trainers.algorithms.configs.modules.DreamerV3ImageEncoderConfig.html#torchrl.trainers.algorithms.configs.modules.DreamerV3ImageEncoderConfig)

forward(*image: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3ImageEncoder.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

output_features(*image_shape: tuple[int, int, int]*) → int[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3ImageEncoder.output_features)

Return the flattened feature count for a `(C, H, W)` image shape.