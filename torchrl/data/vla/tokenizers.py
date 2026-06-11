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

__all__ = [
    "ActionTokenizerBase",
    "UniformActionTokenizer",
    "VocabTailActionTokenizer",
]


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
        """Build from the ``action_low``/``action_high`` of a :class:`~torchrl.data.vla.RobotDatasetMetadata`."""
        if metadata.action_low is None or metadata.action_high is None:
            raise ValueError(
                f"metadata {metadata.dataset_id!r} has no action bounds "
                "(set action_low/action_high) for uniform tokenization."
            )
        return cls(num_bins, low=metadata.action_low, high=metadata.action_high)


class VocabTailActionTokenizer(ActionTokenizerBase):
    r"""OpenVLA-style vocab-tail action tokenizer.

    OpenVLA (`arXiv:2406.09246 <https://arxiv.org/abs/2406.09246>`_)
    discretizes each normalized action dimension over the *edges* of
    ``num_bins`` uniform bins spanning ``[-1, 1]`` and writes the result into
    the last ``num_bins`` ids of the language-model vocabulary:
    ``full_token_id = vocab_size - digitize(action)``. Decoding maps a token
    back to the corresponding bin center (there are ``num_bins - 1`` centers).
    This tokenizer reproduces that exact mapping, with two id conventions:

    - **window ids** (default, ``full_vocab_size=None``): ids in
      ``[0, num_bins)`` -- the offset of the token inside the vocab-tail
      window, ``window_id = num_bins - digitize(action)``. This is the
      convention of a token-head VLA policy emitting a ``num_bins``-way
      categorical per action dimension (e.g.
      :class:`~torchrl.modules.vla.VLAWrapperBase` with
      ``vocab_size=num_bins``).
    - **full ids**: pass ``full_vocab_size`` (e.g. ``32000`` for LLaMA-2) to
      use raw language-model token ids,
      ``full_id = full_vocab_size - digitize(action)``.

    Optionally, dataset statistics (the ``norm_stats`` shipped with OpenVLA
    checkpoints) un-normalize decoded actions to the environment's action
    space -- and normalize actions before encoding -- via the affine q01/q99
    map ``a_env = 0.5 * (a + 1) * (q99 - q01) + q01`` applied to the
    dimensions selected by ``mask`` (the gripper dimension is typically
    excluded). See :meth:`from_norm_stats`.

    Args:
        num_bins (int): number of bin edges per action dimension (the OpenVLA
            convention; there are ``num_bins - 1`` bin centers). Defaults to
            ``256``.

    Keyword Args:
        full_vocab_size (int, optional): if provided, tokens are raw
            language-model ids in ``[full_vocab_size - num_bins,
            full_vocab_size)`` instead of window offsets. Defaults to
            ``None``.
        norm_low (torch.Tensor, optional): per-dimension lower statistics
            (``q01``) for un-normalization. Defaults to ``None`` (no
            normalization; actions live in ``[-1, 1]``).
        norm_high (torch.Tensor, optional): per-dimension upper statistics
            (``q99``).
        norm_mask (torch.Tensor, optional): boolean mask of the dimensions to
            (un-)normalize; unmasked dimensions pass through. Defaults to all
            ``True`` when statistics are given.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import VocabTailActionTokenizer
        >>> tok = VocabTailActionTokenizer(256)
        >>> tokens = tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
        >>> tokens
        tensor([255, 128,   0])
        >>> tok.decode(tokens)
        tensor([-0.9961,  0.0000,  0.9961])
        >>> # full LM-vocabulary ids (LLaMA-2)
        >>> tok = VocabTailActionTokenizer(256, full_vocab_size=32000)
        >>> tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
        tensor([31999, 31872, 31744])
        >>> tok.vocab_size
        32000

    .. seealso:: :class:`~torchrl.data.vla.UniformActionTokenizer` for the
        plain bin-index codec used by toy token policies.
    """

    def __init__(
        self,
        num_bins: int = 256,
        *,
        full_vocab_size: int | None = None,
        norm_low: torch.Tensor | None = None,
        norm_high: torch.Tensor | None = None,
        norm_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}.")
        if full_vocab_size is not None and full_vocab_size < num_bins:
            raise ValueError(
                f"full_vocab_size ({full_vocab_size}) must be at least "
                f"num_bins ({num_bins})."
            )
        if (norm_low is None) != (norm_high is None):
            raise ValueError("norm_low and norm_high must be provided together.")
        self.num_bins = int(num_bins)
        self.full_vocab_size = (
            int(full_vocab_size) if full_vocab_size is not None else None
        )
        bins = torch.linspace(-1.0, 1.0, num_bins)
        self.register_buffer("bins", bins)
        self.register_buffer("bin_centers", (bins[:-1] + bins[1:]) / 2.0)
        if norm_low is not None:
            norm_low = torch.as_tensor(norm_low, dtype=torch.float32)
            norm_high = torch.as_tensor(norm_high, dtype=torch.float32)
            if norm_mask is None:
                norm_mask = torch.ones_like(norm_low, dtype=torch.bool)
            else:
                norm_mask = torch.as_tensor(norm_mask, dtype=torch.bool)
            self.register_buffer("norm_low", norm_low)
            self.register_buffer("norm_high", norm_high)
            self.register_buffer("norm_mask", norm_mask)
        else:
            self.norm_low = self.norm_high = self.norm_mask = None

    @property
    def vocab_size(self) -> int:
        if self.full_vocab_size is not None:
            return self.full_vocab_size
        return self.num_bins

    def _digitize(self, actions: torch.Tensor) -> torch.Tensor:
        # exact torch port of np.digitize(clip(a, -1, 1), bins): index of the
        # first bin edge strictly greater than the value, i.e. in [1, num_bins]
        actions = actions.clamp(-1.0, 1.0)
        return torch.bucketize(actions, self.bins, right=True)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        if self.norm_low is not None:
            scale = (self.norm_high - self.norm_low).clamp_min(1e-8)
            normalized = 2.0 * (actions - self.norm_low) / scale - 1.0
            actions = torch.where(self.norm_mask, normalized, actions)
        digitized = self._digitize(actions)
        if self.full_vocab_size is not None:
            return self.full_vocab_size - digitized
        return self.num_bins - digitized

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.full_vocab_size is not None:
            digitized = self.full_vocab_size - tokens
        else:
            digitized = self.num_bins - tokens
        index = (digitized - 1).clamp(0, self.bin_centers.shape[0] - 1)
        actions = self.bin_centers[index]
        if self.norm_low is not None:
            unnormalized = (
                0.5 * (actions + 1.0) * (self.norm_high - self.norm_low) + self.norm_low
            )
            actions = torch.where(self.norm_mask, unnormalized, actions)
        return actions

    @classmethod
    def from_norm_stats(
        cls,
        norm_stats: dict,
        unnorm_key: str,
        *,
        num_bins: int = 256,
        full_vocab_size: int | None = None,
    ) -> VocabTailActionTokenizer:
        """Build from the ``norm_stats`` dictionary of an OpenVLA checkpoint.

        Args:
            norm_stats (dict): the checkpoint's normalization statistics
                (``model.norm_stats``), mapping dataset keys to
                ``{"action": {"q01": ..., "q99": ..., "mask": ...}}``.
            unnorm_key (str): the dataset key to use (e.g.
                ``"libero_spatial_no_noops"``).

        Keyword Args:
            num_bins (int, optional): number of bin edges. Defaults to ``256``.
            full_vocab_size (int, optional): see the class docstring.
        """
        if unnorm_key not in norm_stats:
            raise KeyError(
                f"unnorm_key {unnorm_key!r} not found in norm_stats; available "
                f"keys: {sorted(norm_stats)}."
            )
        stats = norm_stats[unnorm_key]["action"]
        mask = stats.get("mask")
        return cls(
            num_bins,
            full_vocab_size=full_vocab_size,
            norm_low=torch.as_tensor(stats["q01"], dtype=torch.float32),
            norm_high=torch.as_tensor(stats["q99"], dtype=torch.float32),
            norm_mask=torch.as_tensor(mask, dtype=torch.bool)
            if mask is not None
            else None,
        )
