# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Action tokenizers for autoregressive (token) VLA policies."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torchrl.data.vla.metadata import RobotDatasetMetadata

__all__ = ["ActionTokenizerBase", "UniformActionTokenizer"]


class ActionTokenizerBase(nn.Module):
    """Base class for action tokenizers.

    An action tokenizer maps continuous actions to discrete token ids and back,
    so that autoregressive (RT-2 / OpenVLA-style) VLA policies can emit actions
    through a language-model head and be trained with token cross-entropy.

    A tokenizer operates element-wise over the trailing action dimension, so it
    works unchanged on per-step actions ``[*B, action_dim]`` and on action
    chunks ``[*B, T, chunk, action_dim]``.

    Subclasses implement :meth:`encode`, :meth:`decode` and the
    :attr:`vocab_size` property.
    """

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Map continuous actions ``[..., action_dim]`` to token ids (``long``)."""
        raise NotImplementedError

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map token ids back to continuous actions ``[..., action_dim]``."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Number of distinct token ids the tokenizer can emit per position."""
        raise NotImplementedError


class UniformActionTokenizer(ActionTokenizerBase):
    """Per-dimension uniform-bin action tokenizer (RT-2 / OpenVLA style).

    Each action dimension is discretized into ``num_bins`` equal-width bins over
    ``[low, high]``; :meth:`encode` returns the bin index and :meth:`decode`
    returns the bin center. The round-trip is lossy with error bounded by half a
    bin width, ``(high - low) / (2 * num_bins)``.

    Args:
        num_bins (int): number of bins per action dimension.

    Keyword Args:
        low (float or torch.Tensor): per-dimension lower bound. Actions are
            clamped to ``[low, high]`` before binning.
        high (float or torch.Tensor): per-dimension upper bound.
        action_dim (int, optional): action dimensionality. Required only when
            ``low``/``high`` are scalars and you want a per-dimension shape.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import UniformActionTokenizer
        >>> tok = UniformActionTokenizer(256, low=-1.0, high=1.0)
        >>> tokens = tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
        >>> tokens
        tensor([  0, 128, 255])
        >>> torch.allclose(tok.decode(tokens), torch.tensor([-0.998, 0.002, 0.998]), atol=1e-2)
        True
        >>> tok.vocab_size
        256

    .. seealso:: :class:`~torchrl.data.vla.RobotDatasetMetadata` carries the
        ``action_low``/``action_high`` bounds used by :meth:`from_metadata`.
    """

    def __init__(
        self,
        num_bins: int,
        *,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
        low = torch.as_tensor(low, dtype=torch.float32)
        high = torch.as_tensor(high, dtype=torch.float32)
        if action_dim is not None:
            # Materialize scalar bounds to per-dimension buffers. This is purely
            # for shape/introspection (``action_dim``); scalar bounds already
            # broadcast correctly in encode/decode without it.
            if low.ndim == 0:
                low = low.repeat(action_dim)
            if high.ndim == 0:
                high = high.repeat(action_dim)
        if low.shape != high.shape:
            raise ValueError(
                f"low and high must have the same shape, got {tuple(low.shape)} "
                f"and {tuple(high.shape)}."
            )
        if not (high > low).all():
            raise ValueError(
                "high must be strictly greater than low for every dimension."
            )
        self.num_bins = int(num_bins)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    @property
    def vocab_size(self) -> int:
        return self.num_bins

    @property
    def action_dim(self) -> int | None:
        """The per-dimension action size, or ``None`` for scalar bounds."""
        return self.low.shape[-1] if self.low.ndim else None

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        scaled = (actions - self.low) / (self.high - self.low)
        tokens = (scaled * self.num_bins).floor().long()
        return tokens.clamp_(0, self.num_bins - 1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        centers = (tokens.to(self.low.dtype) + 0.5) / self.num_bins
        return self.low + centers * (self.high - self.low)

    @classmethod
    def from_metadata(
        cls, metadata: RobotDatasetMetadata, num_bins: int
    ) -> UniformActionTokenizer:
        """Build from the ``action_low``/``action_high`` of a
        :class:`~torchrl.data.vla.RobotDatasetMetadata`."""
        if metadata.action_low is None or metadata.action_high is None:
            raise ValueError(
                f"metadata {metadata.dataset_id!r} has no action bounds "
                "(set action_low/action_high) for uniform tokenization."
            )
        return cls(num_bins, low=metadata.action_low, high=metadata.action_high)
