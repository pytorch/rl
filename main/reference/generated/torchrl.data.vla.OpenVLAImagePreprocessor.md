# OpenVLAImagePreprocessor

*class*torchrl.data.vla.OpenVLAImagePreprocessor(***, *size: int = 224*, *jpeg_quality: int = 95*, *center_crop: bool = False*, *backend: Literal['torchvision', 'pil', 'tensorflow'] = 'torchvision'*, *mean: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | Sequence[float] | Sequence[Sequence[float]] | None = None*, *std: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | Sequence[float] | Sequence[Sequence[float]] | None = None*)[[source]](../../_modules/torchrl/data/vla/preprocessing.html#OpenVLAImagePreprocessor)

OpenVLA-style image resize, JPEG round-trip and optional center crop.

The `"tensorflow"` backend mirrors the OpenVLA-OFT evaluation path:
JPEG encode/decode at the requested quality, resize with Lanczos3,
optionally apply a 0.9-area center crop, and resize back. The default
`"torchvision"` backend keeps data as tensors and uses
`torchvision.io` JPEG codecs; `"pil"` is a lightweight debugging
backend.

Parameters:

- **size** (*int*) - Square output size. Defaults to `224`.
- **jpeg_quality** (*int*) - JPEG quality. Defaults to `95`.
- **center_crop** (*bool*) - Whether to apply the OpenVLA 0.9-area center crop.
Defaults to `False`.
- **backend** (*str*) - `"torchvision"`, `"pil"` or `"tensorflow"`.
Defaults to `"torchvision"`.
- **mean** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*|**sequence**,**optional*) - Per-channel normalization mean.
A two-dimensional sequence applies multiple normalizations to the
same image and concatenates the results along the channel axis,
as required by fused OpenVLA vision backbones.
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