# OpenVLAImagePreprocessor

*class*torchrl.data.vla.OpenVLAImagePreprocessor(***, *size: int = 224*, *jpeg_quality: int = 95*, *center_crop: bool = False*, *backend: Literal['torchvision', 'pil'] = 'torchvision'*, *mean: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | list[float] | tuple[float, ...] | None = None*, *std: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | list[float] | tuple[float, ...] | None = None*)[[source]](../../_modules/torchrl/data/vla/preprocessing.html#OpenVLAImagePreprocessor)

OpenVLA-style image resize, JPEG round-trip and optional center crop.

The operation order mirrors the OpenVLA-OFT evaluation path: resize to a
square image, JPEG encode/decode at the requested quality, optionally apply
a 0.9-area center crop, and resize back. The default `"torchvision"`
backend keeps data as tensors and uses `torchvision.io` JPEG codecs;
`"pil"` is the reference/debugging backend.

Parameters:

- **size** (*int*) - Square output size. Defaults to `224`.
- **jpeg_quality** (*int*) - JPEG quality. Defaults to `95`.
- **center_crop** (*bool*) - Whether to apply the OpenVLA 0.9-area center crop.
Defaults to `False`.
- **backend** (*str*) - `"torchvision"` or `"pil"`. Defaults to
`"torchvision"`.
- **mean** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*|**sequence**,**optional*) - Per-channel normalization mean.
- **std** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*|**sequence**,**optional*) - Per-channel normalization std.

Note

Floating-point inputs are ambiguous: this helper treats float images with
maximum value at most `1` as normalized `[0, 1]` data and rescales
them to uint8; other float images are interpreted as `[0, 255]` data.

Examples

```
>>> import torch
>>> from torchrl.data.vla import OpenVLAImagePreprocessor
>>> proc = OpenVLAImagePreprocessor(backend="pil")
>>> out = proc(torch.zeros(2, 3, 32, 32, dtype=torch.uint8))
>>> out.shape
torch.Size([2, 3, 224, 224])
```