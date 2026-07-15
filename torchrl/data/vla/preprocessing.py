# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Image preprocessing helpers for Vision-Language-Action policies."""
from __future__ import annotations

import importlib.util
import io
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from torchrl._utils import implement_for, is_compiling

_has_pil = importlib.util.find_spec("PIL") is not None
_has_tensorflow = importlib.util.find_spec("tensorflow") is not None
_has_torchvision = importlib.util.find_spec("torchvision") is not None

__all__ = ["OpenVLAImagePreprocessor"]

_CROP_AREA_SCALE = 0.9
_CROP_LINEAR_SCALE = _CROP_AREA_SCALE**0.5
_LANCZOS_RADIUS = 3
_REFERENCE_RESIZE_BATCH_SIZE = 8


def _make_lanczos3_spans(
    input_size: int,
    output_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the sampling spans used by TensorFlow's ScaleAndTranslate op."""
    scale = torch.tensor(
        output_size, dtype=torch.float32, device=device
    ) / torch.tensor(input_size, dtype=torch.float32, device=device)
    inv_scale = scale.reciprocal()
    kernel_scale = inv_scale.clamp_min(1.0)
    radius = torch.tensor(_LANCZOS_RADIUS, dtype=torch.float32, device=device)
    # ceil(3 * max(input_size / output_size, 1)) using integer arithmetic.
    # The extra sample on each side matches TensorFlow's span allocation.
    kernel_radius = max(
        (_LANCZOS_RADIUS * input_size + output_size - 1) // output_size,
        _LANCZOS_RADIUS,
    )
    span_size = min(
        2 * kernel_radius + 1,
        input_size,
    )
    samples = (
        torch.arange(output_size, dtype=torch.float32, device=device) + 0.5
    ) * inv_scale
    span_starts = torch.ceil(samples - radius * kernel_scale - 0.5)
    span_starts = span_starts.to(torch.long).clamp(0, input_size - 1)
    span_ends = torch.floor(samples + radius * kernel_scale - 0.5)
    span_ends = span_ends.to(torch.long).clamp(0, input_size - 1) + 1

    span_offsets = torch.arange(span_size, device=device)
    source_indices = span_starts.unsqueeze(1) + span_offsets
    valid = source_indices < span_ends.unsqueeze(1)
    indices = source_indices.masked_fill(~valid, 0)

    kernel_positions = (
        (source_indices.to(torch.float32) + 0.5 - samples.unsqueeze(1)) / kernel_scale
    ).abs()
    near_zero = kernel_positions <= 1e-3
    safe_positions = kernel_positions.masked_fill(near_zero, 1.0)
    pi = torch.tensor(3.14159265359, dtype=torch.float32, device=device)
    weights = (
        radius
        * torch.sin(pi * safe_positions)
        * torch.sin(pi * safe_positions / radius)
        / (pi * pi * safe_positions * safe_positions)
    )
    weights = torch.where(near_zero, 1.0, weights)
    weights = weights.masked_fill(~valid | (kernel_positions > radius), 0.0)
    total_weight = weights.sum(-1, keepdim=True)
    normalizable = total_weight.abs() >= 1000.0 * torch.finfo(torch.float32).tiny
    normalizer = torch.where(normalizable, total_weight.reciprocal(), 1.0)
    return indices, weights * normalizer


@lru_cache(maxsize=32)
def _cached_lanczos3_spans(
    input_size: int,
    output_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _make_lanczos3_spans(input_size, output_size, device)


def _lanczos3_spans(
    input_size: int,
    output_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device("cpu")
    if is_compiling():
        return _make_lanczos3_spans(input_size, output_size, device)
    return _cached_lanczos3_spans(input_size, output_size, device)


def _apply_spans(
    values: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    output_size, span_size = indices.shape
    sampled = values.index_select(-1, indices.flatten()).reshape(
        *values.shape[:-1], output_size, span_size
    )
    weight_shape = (1,) * (values.ndim - 1) + (output_size, span_size)
    return (sampled * weights.reshape(weight_shape)).sum(-1)


def _lanczos3_resize(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize NCHW float images with TensorFlow-compatible Lanczos3 spans."""
    output_height, output_width = size
    height_indices, height_weights = _lanczos3_spans(
        images.shape[-2], output_height, images.device
    )
    width_indices, width_weights = _lanczos3_spans(
        images.shape[-1], output_width, images.device
    )

    resized = _apply_spans(images, width_indices, width_weights)
    resized = _apply_spans(
        resized.transpose(-2, -1), height_indices, height_weights
    ).transpose(-2, -1)
    return resized


def _make_crop_axis(
    input_size: int, output_size: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    crop_size = torch.tensor(
        _CROP_AREA_SCALE, dtype=torch.float32, device=device
    ).sqrt()
    offset = (1.0 - crop_size) / 2.0
    end = offset + crop_size
    if output_size > 1:
        scale = (end - offset) * (input_size - 1) / (output_size - 1)
        coordinates = (
            offset * (input_size - 1)
            + torch.arange(output_size, dtype=torch.float32, device=device) * scale
        )
    else:
        coordinates = (0.5 * (offset + end) * (input_size - 1)).unsqueeze(0)
    lower = coordinates.floor().to(torch.long)
    upper = coordinates.ceil().to(torch.long)
    return lower, upper, coordinates - lower.to(torch.float32)


@lru_cache(maxsize=32)
def _cached_crop_axis(
    input_size: int, output_size: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return _make_crop_axis(input_size, output_size, device)


def _crop_axis(
    input_size: int,
    output_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    if device is None:
        device = torch.device("cpu")
    if is_compiling():
        return _make_crop_axis(input_size, output_size, device)
    return _cached_crop_axis(input_size, output_size, device)


def _fractional_center_crop(
    images: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Apply TensorFlow CropAndResize's 0.9-area centered bilinear crop."""
    output_height, output_width = size
    top, bottom, y_lerp = _crop_axis(images.shape[-2], output_height, images.device)
    left, right, x_lerp = _crop_axis(images.shape[-1], output_width, images.device)

    values = images.to(torch.float32).div(255.0)
    left_values = values.index_select(-1, left)
    right_values = values.index_select(-1, right)
    x_lerp = x_lerp.reshape((1,) * (values.ndim - 1) + (output_width,))
    values = left_values + (right_values - left_values) * x_lerp

    top_values = values.index_select(-2, top)
    bottom_values = values.index_select(-2, bottom)
    y_lerp = y_lerp.reshape((1,) * (values.ndim - 2) + (output_height, 1))
    values = top_values + (bottom_values - top_values) * y_lerp
    # TensorFlow's float-to-uint8 conversion multiplies by max + 0.5 before
    # truncating, rather than rounding after multiplying by max.
    return values.mul_(255.5).clamp_(0, 255).to(torch.uint8)


@implement_for("torchvision", None, "0.20")
def _torchvision_jpeg_roundtrip(
    images: torch.Tensor, jpeg_quality: int, device: torch.device
) -> torch.Tensor:
    from torchvision.io import decode_jpeg, encode_jpeg, ImageReadMode

    encoded = [
        encode_jpeg(image.cpu(), quality=jpeg_quality) for image in images.unbind(0)
    ]
    decoded = [
        decode_jpeg(image, mode=ImageReadMode.RGB, device=device) for image in encoded
    ]
    return torch.stack(decoded, 0)


@implement_for("torchvision", "0.20")
def _torchvision_jpeg_roundtrip(  # noqa: F811
    images: torch.Tensor, jpeg_quality: int, device: torch.device
) -> torch.Tensor:
    from torchvision.io import decode_jpeg, encode_jpeg, ImageReadMode

    encoded = encode_jpeg(list(images.unbind(0)), quality=jpeg_quality)
    decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB, device=device)
    if isinstance(decoded, list):
        decoded = torch.stack(decoded, 0)
    return decoded


class OpenVLAImagePreprocessor:
    """OpenVLA-style image resize, JPEG round-trip and optional center crop.

    The ``"tensorflow"`` backend mirrors the OpenVLA-OFT evaluation path:
    JPEG encode/decode at the requested quality, resize with Lanczos3,
    optionally apply a 0.9-area center crop, and resize back. The
    ``"torch_reference"`` backend follows the same order and interpolation
    semantics using PyTorch and ``torchvision`` only, and is the default. The
    ``"torchvision"`` backend keeps data as tensors but uses a faster bicubic
    path; ``"pil"`` is a lightweight debugging backend.

    Args:
        size (int): Square output size. Defaults to ``224``.
        jpeg_quality (int): JPEG quality. Defaults to ``95``.
        center_crop (bool): Whether to apply the OpenVLA 0.9-area center crop.
            Defaults to ``False``.
        backend (str): ``"torchvision"``, ``"torch_reference"``, ``"pil"``
            or ``"tensorflow"``. Defaults to ``"torch_reference"``.
        mean (torch.Tensor | sequence, optional): Per-channel normalization mean.
            A two-dimensional sequence applies multiple normalizations to the
            same image and concatenates the results along the channel axis,
            as required by fused OpenVLA vision backbones.
        std (torch.Tensor | sequence, optional): Per-channel normalization std.

    .. note::
        Floating-point inputs are ambiguous: this helper treats float images with
        maximum value at most ``1`` as normalized ``[0, 1]`` data and rescales
        them to uint8; other float images are interpreted as ``[0, 255]`` data.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import OpenVLAImagePreprocessor
        >>> proc = OpenVLAImagePreprocessor(backend="pil")
        >>> out = proc(torch.zeros(2, 3, 32, 32, dtype=torch.uint8))
        >>> out.shape
        torch.Size([2, 3, 224, 224])
    """

    def __init__(
        self,
        *,
        size: int = 224,
        jpeg_quality: int = 95,
        center_crop: bool = False,
        backend: Literal[
            "torchvision", "torch_reference", "pil", "tensorflow"
        ] = "torch_reference",
        mean: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        std: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}.")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"jpeg_quality must be between 1 and 100, got {jpeg_quality}."
            )
        if backend not in ("torchvision", "torch_reference", "pil", "tensorflow"):
            raise ValueError(
                "backend must be 'torchvision', 'torch_reference', 'pil' or "
                f"'tensorflow', got {backend!r}."
            )
        if backend in ("torchvision", "torch_reference") and not _has_torchvision:
            raise ImportError(
                f"OpenVLAImagePreprocessor backend={backend!r} requires torchvision. "
                "Install the TorchRL vla extra or pass backend='pil'."
            )
        if backend == "pil" and not _has_pil:
            raise ImportError("OpenVLAImagePreprocessor backend='pil' requires Pillow.")
        if backend == "tensorflow" and not _has_tensorflow:
            raise ImportError(
                "OpenVLAImagePreprocessor backend='tensorflow' requires TensorFlow."
            )
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together.")
        self.size = int(size)
        self.jpeg_quality = int(jpeg_quality)
        self.center_crop = bool(center_crop)
        self.backend = backend
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float32)
        if self.mean is not None:
            if self.mean.shape != self.std.shape:
                raise ValueError(
                    "mean and std must have matching shapes, got "
                    f"{tuple(self.mean.shape)} and {tuple(self.std.shape)}."
                )
            if self.mean.ndim not in (1, 2) or self.mean.shape[-1] != 3:
                raise ValueError(
                    "mean and std must have shape [3] or [N, 3], got "
                    f"{tuple(self.mean.shape)}."
                )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess one image ``[C, H, W]`` or a batch ``[..., C, H, W]``."""
        if images.ndim < 3:
            raise ValueError(
                f"images must have shape [..., C, H, W], got {tuple(images.shape)}."
            )
        if images.shape[-3] not in (1, 3):
            raise ValueError(
                "images must have one or three channels in the third-to-last "
                f"dimension, got shape {tuple(images.shape)}."
            )
        original_shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        flat = self._to_uint8(flat)
        if self.backend == "torchvision":
            processed = self._torchvision(flat)
        elif self.backend == "torch_reference":
            processed = self._torch_reference(flat)
        elif self.backend == "tensorflow":
            processed = self._tensorflow(flat)
        else:
            processed = self._pil(flat)
        processed = processed.reshape(*original_shape, *processed.shape[-3:])
        if self.mean is not None:
            processed = processed.to(torch.float32).div_(255.0)
            means = self.mean.reshape(-1, self.mean.shape[-1]).to(
                device=processed.device, dtype=processed.dtype
            )
            stds = self.std.reshape(-1, self.std.shape[-1]).to(
                device=processed.device, dtype=processed.dtype
            )
            view_shape = *((1,) * (processed.ndim - 3)), -1, 1, 1
            processed = torch.cat(
                [
                    processed.sub(mean.view(view_shape)).div(std.view(view_shape))
                    for mean, std in zip(means, stds, strict=True)
                ],
                dim=-3,
            )
        return processed

    @staticmethod
    def _to_uint8(images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            return images.contiguous()
        if images.is_floating_point():
            scale = 1.0
            # Float inputs are accepted either as normalized [0, 1] images or
            # as [0, 255] images; use the max value to disambiguate.
            if images.numel() and float(images.detach().max()) <= 1.0:
                scale = 255.0
            return images.mul(scale).clamp(0, 255).to(torch.uint8).contiguous()
        return images.clamp(0, 255).to(torch.uint8).contiguous()

    def _resize(self, images: torch.Tensor, size: int | None = None) -> torch.Tensor:
        size = self.size if size is None else int(size)
        if images.shape[-2:] == (size, size):
            return images
        resized = F.interpolate(
            images.to(torch.float32),
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return resized.round_().clamp_(0, 255).to(torch.uint8)

    def _center_crop(self, images: torch.Tensor) -> torch.Tensor:
        if not self.center_crop:
            return images
        height, width = images.shape[-2:]
        crop_h = int(round(height * _CROP_LINEAR_SCALE))
        crop_w = int(round(width * _CROP_LINEAR_SCALE))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return images[..., top : top + crop_h, left : left + crop_w]

    def _torchvision(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        images = self._resize(images)
        decoded = _torchvision_jpeg_roundtrip(images, self.jpeg_quality, device)
        decoded = self._center_crop(decoded)
        decoded = self._resize(decoded)
        return decoded

    def _torch_reference(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        # torchvision's reference JPEG codec is CPU-only. Keep the subsequent
        # interpolation on its decoded CPU output and restore the caller's
        # device only after preprocessing is complete.
        images = images.cpu()
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1).contiguous()
        decoded = _torchvision_jpeg_roundtrip(
            images, self.jpeg_quality, torch.device("cpu")
        )
        resized = []
        for chunk in decoded.split(_REFERENCE_RESIZE_BATCH_SIZE):
            chunk = _lanczos3_resize(chunk.to(torch.float32), (self.size, self.size))
            resized.append(chunk.round_().clamp_(0, 255).to(torch.uint8))
        decoded = torch.cat(resized, 0)
        if self.center_crop:
            decoded = _fractional_center_crop(decoded, (self.size, self.size))
        return decoded.to(device)

    def _tensorflow(self, images: torch.Tensor) -> torch.Tensor:
        import tensorflow as tf

        out = []
        resize_size = (self.size, self.size)
        for image in images.cpu():
            array = image.permute(1, 2, 0).numpy().astype("uint8")
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            tf_image = tf.convert_to_tensor(array)
            tf_image = tf.image.encode_jpeg(tf_image, quality=self.jpeg_quality)
            tf_image = tf.io.decode_image(
                tf_image, expand_animations=False, dtype=tf.uint8
            )
            tf_image = tf.image.resize(
                tf_image, resize_size, method="lanczos3", antialias=True
            )
            tf_image = tf.cast(tf.clip_by_value(tf.round(tf_image), 0, 255), tf.uint8)
            if self.center_crop:
                expanded = tf.expand_dims(
                    tf.image.convert_image_dtype(tf_image, tf.float32), axis=0
                )
                crop_size = tf.reshape(
                    tf.clip_by_value(tf.sqrt(tf.constant(_CROP_AREA_SCALE)), 0, 1),
                    shape=(1,),
                )
                offsets = (1 - crop_size) / 2
                boxes = tf.stack(
                    [
                        offsets,
                        offsets,
                        offsets + crop_size,
                        offsets + crop_size,
                    ],
                    axis=1,
                )
                tf_image = tf.image.crop_and_resize(
                    expanded, boxes, tf.range(1), resize_size
                )[0]
                tf_image = tf.clip_by_value(tf_image, 0, 1)
                tf_image = tf.image.convert_image_dtype(
                    tf_image, tf.uint8, saturate=True
                )
            out.append(
                torch.from_numpy(
                    np.asarray(tf_image.numpy(), dtype="uint8").copy()
                ).permute(2, 0, 1)
            )
        return torch.stack(out, 0).to(images.device)

    def _pil(self, images: torch.Tensor) -> torch.Tensor:
        from PIL import Image

        out = []
        for image in images.cpu():
            array = image.permute(1, 2, 0).numpy().astype("uint8")
            pil = Image.fromarray(array.squeeze(-1) if array.shape[-1] == 1 else array)
            pil = pil.convert("RGB")
            if pil.size != (self.size, self.size):
                pil = pil.resize((self.size, self.size), Image.LANCZOS)
            buffer = io.BytesIO()
            pil.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            pil = Image.open(buffer).convert("RGB")
            if self.center_crop:
                width, height = pil.size
                crop_w = int(round(width * _CROP_LINEAR_SCALE))
                crop_h = int(round(height * _CROP_LINEAR_SCALE))
                left = (width - crop_w) // 2
                top = (height - crop_h) // 2
                pil = pil.crop((left, top, left + crop_w, top + crop_h)).resize(
                    (width, height), Image.LANCZOS
                )
            out.append(
                torch.from_numpy(np.asarray(pil, dtype="uint8").copy()).permute(2, 0, 1)
            )
        return torch.stack(out, 0).to(images.device)
