# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Structured TensorClass containers for Vision-Language-Action data."""
from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import TensorClass

__all__ = ["VLAAction", "VLAImages", "VLAObservation"]


class VLAImages(TensorClass["nocast"]):
    """Container for VLA image observations.

    Args:
        image (torch.Tensor | None): Primary camera image tensor.
        wrist_image (torch.Tensor | None): Optional wrist-camera image tensor.
        extra (TensorDictBase | None): Optional additional camera/features data.
        padded (bool | None): Whether images were padded to a common shape.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import VLAImages
        >>> images = VLAImages(image=torch.zeros(2, 3, 16, 16), batch_size=[2])
        >>> images.image.shape
        torch.Size([2, 3, 16, 16])
    """

    image: torch.Tensor | None = None
    wrist_image: torch.Tensor | None = None
    extra: TensorDictBase | None = None
    padded: bool | None = None


class VLAObservation(TensorClass["nocast"]):
    """Container for VLA observations.

    Args:
        images (VLAImages | None): Structured camera observations.
        state (torch.Tensor | None): Optional proprioceptive state.
        instruction (object | None): Raw or tokenized language instruction.
        preprocessed (TensorDictBase | None): Backend-ready model inputs.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import VLAImages, VLAObservation
        >>> obs = VLAObservation(
        ...     images=VLAImages(image=torch.zeros(2, 3, 16, 16), batch_size=[2]),
        ...     state=torch.zeros(2, 5),
        ...     instruction=["pick", "place"],
        ...     batch_size=[2],
        ... )
        >>> obs.images.image.shape
        torch.Size([2, 3, 16, 16])
    """

    images: VLAImages | None = None
    state: torch.Tensor | None = None
    instruction: object | None = None
    preprocessed: TensorDictBase | None = None


class VLAAction(TensorClass["nocast", "shadow"]):
    """Container for VLA policy outputs.

    Args:
        chunk (torch.Tensor | None): Continuous action chunk.
        tokens (torch.Tensor | None): Action-token ids in the policy vocabulary.
        raw_tokens (torch.Tensor | None): Backend-native token ids when they differ
            from action-token window ids.
        logits (torch.Tensor | None): Token logits with a trailing vocabulary dim.
        log_probs (torch.Tensor | None): Log-probabilities of the selected tokens.
        mask (torch.Tensor | None): Optional valid-token/action mask.
        padded (bool | None): Whether variable-length outputs were padded.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import VLAAction
        >>> action = VLAAction(chunk=torch.zeros(2, 4, 7), batch_size=[2])
        >>> action.chunk.shape
        torch.Size([2, 4, 7])
    """

    chunk: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    raw_tokens: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    padded: bool | None = None


def _get_action_chunk(action: VLAAction) -> torch.Tensor | None:
    return action._tensordict.get("chunk", None)


def _set_action_chunk(action: VLAAction, value: torch.Tensor | None) -> None:
    action._tensordict.set("chunk", value)


# ``chunk`` also names a TensorDictBase method. ``shadow`` lets the field be
# stored, but the generated descriptor would otherwise expose the method rather
# than the action tensor. Re-install the descriptor explicitly so
# ``vla_action.chunk`` and ``td["vla_action", "chunk"]`` stay equivalent.
VLAAction.chunk = property(_get_action_chunk, _set_action_chunk)
