# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Literal

import torch
from tensordict.utils import expand_as_right
from torch import Tensor

__all__ = [
    "cat_frames",
    "rgb_to_grayscale",
]


# copied from torchvision
def _get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError(f"Input ndim should be 2 or more. Got {img.ndim}")


def _assert_channels(img: Tensor, permitted: list[int]) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError(
            f"Input image tensor permitted channel values are {permitted}, but found "
            f"{c} (full shape: {img.shape})"
        )


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    """Turns an RGB image into grayscale."""
    if img.ndim < 3:
        raise TypeError(
            "Input image tensor should have at least 3 dimensions, but found"
            "{}".format(img.ndim)
        )
    _assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def _apply_same_padding(dim: int, data: Tensor, done_mask: Tensor) -> Tensor:
    """Replaces frames flagged by ``done_mask`` with the first valid frame of the window.

    This implements the ``padding="same"`` semantics of :func:`cat_frames`: within
    each sliding window, the entries that reach across a trajectory boundary
    (marked by ``done_mask``) are overwritten with the earliest in-trajectory
    frame of that window. ``data`` is the permuted, windowed tensor (the window
    axis already moved to ``data.ndim + dim - 1``); ``done_mask`` is the
    un-permuted boolean mask of shape ``(*batch, time, 1, N)`` (the singleton is
    the done "feature" dim).
    """
    d = data.ndim + dim - 1
    N = done_mask.shape[-1]
    # The out-of-trajectory slots always form a prefix of the window (the
    # oldest frames), so slot ``j`` reads slot ``max(j, num_padded)``: its own
    # value once past the prefix, the first in-trajectory frame otherwise.
    num_padded = done_mask.sum(dim=-1)
    index = torch.maximum(torch.arange(N, device=data.device), num_padded).clamp_max_(
        N - 1
    )
    data = data.movedim(d, -1)
    index = index.reshape(*index.shape[:-1], *(1,) * (data.ndim - index.ndim), N)
    return data.gather(-1, index.expand(data.shape)).movedim(-1, d)


def _cat_frames_windows(
    tensor: Tensor,
    N: int,
    dim: int,
    *,
    padding: Literal["same", "constant"] = "same",
    padding_value: float = 0.0,
    time_dim: int = -1,
    done_mask: Tensor | None = None,
) -> Tensor:
    """Builds the padded, windowed (but **not** flattened) frame stack.

    This is the shared core of :func:`cat_frames`. It returns the unfolded tensor
    with the window axis placed just before the concatenation axis (``dim``), so
    that a caller can either flatten it (as :func:`cat_frames` does) or run an
    extra fixup before flattening (as :class:`~torchrl.envs.transforms.CatFrames`
    does for ``("next", key)`` inputs).

    ``time_dim`` and ``dim`` are both negative and independent: the sliding
    window moves along ``time_dim`` and the resulting ``N`` frames are
    concatenated along ``dim``. ``done_mask`` (if provided) has its time axis at
    position ``time_dim`` and its window axis appended last, i.e. shape
    ``tensor.shape`` (up to ``time_dim``) followed by a trailing ``N``.
    """
    if dim >= 0:
        raise ValueError(
            "dim must be < 0 to accommodate for tensors of different batch-sizes "
            "(since negative dims are batch invariant)."
        )
    if time_dim >= 0:
        raise ValueError("time_dim must be < 0.")
    # absolute index of the time axis within ``tensor``
    time_pos = tensor.ndim + time_dim
    # pad N-1 frames at the start of the trajectory
    idx = [slice(None)] * time_pos + [0]
    data0 = [torch.full_like(tensor[tuple(idx)], padding_value).unsqueeze(time_pos)] * (
        N - 1
    )
    data = torch.cat(data0 + [tensor], time_pos)
    # unfold along time: appends a trailing window axis of size N
    data = data.unfold(time_pos, N, 1)

    # move the trailing window axis to just before the cat axis ``dim``
    data = data.permute(
        *range(0, data.ndim + dim - 1),
        -1,
        *range(data.ndim + dim - 1, data.ndim - 1),
    )

    if padding != "same":
        if done_mask is not None:
            done_mask_expand = done_mask.view(
                *done_mask.shape[: time_pos + 1],
                *(1,) * (data.ndim - 2 - time_pos),
                done_mask.shape[-1],
            )
            # expand_as_right needs the window axis last, so expand before
            # permuting the window axis into place.
            data_win_last = data.movedim(data.ndim + dim - 1, -1)
            done_mask_expand = expand_as_right(done_mask_expand, data_win_last)
            done_mask_expand = done_mask_expand.movedim(-1, data.ndim + dim - 1)
            data = torch.where(done_mask_expand, padding_value, data)
    else:
        if done_mask is not None:
            data = _apply_same_padding(dim, data, done_mask)
    return data


def cat_frames(
    tensor: Tensor,
    N: int,
    dim: int,
    *,
    padding: Literal["same", "constant"] = "same",
    padding_value: float = 0.0,
    time_dim: int = -1,
    done_mask: Tensor | None = None,
) -> Tensor:
    r"""Stacks a sliding window of ``N`` successive frames along ``dim``.

    This is the pure, stateless core of the
    :class:`~torchrl.envs.transforms.CatFrames` transform (the PyTorch
    ``F.x`` / ``nn.X`` split): :class:`~torchrl.envs.transforms.CatFrames`
    delegates its offline / replay-buffer (contiguous trajectory slice)
    windowing to this function so that the two stay byte-for-byte identical.

    For every position ``t`` along ``time_dim``, the ``N`` frames
    ``[t - N + 1, ..., t]`` are concatenated along ``dim``. The first ``N - 1``
    positions of a trajectory have fewer than ``N`` real frames; the missing
    frames are filled according to ``padding``. This matches the offline
    behavior of :class:`~torchrl.envs.transforms.CatFrames`; see the
    "Examples" of that class for the online (stateful, per-step) usage.

    It was first proposed in "Playing Atari with Deep Reinforcement Learning"
    (https://arxiv.org/abs/1312.5602).

    Args:
        tensor (torch.Tensor): the frames to stack. One of its dimensions
            (``time_dim``) is the time axis along which the sliding window
            moves; ``dim`` is the (channel/feature) axis along which the
            ``N`` frames are concatenated.
        N (int): number of successive frames to concatenate.
        dim (int): the dimension along which the frames are concatenated.
            Must be negative so that it is invariant to leading batch
            dimensions. The size of ``tensor`` along ``dim`` is multiplied by
            ``N`` in the output.

    Keyword Args:
        padding (str, optional): the padding method, one of ``"same"`` or
            ``"constant"``. With ``"same"`` (default) the first real frame of
            the trajectory is repeated; with ``"constant"`` the missing frames
            are filled with ``padding_value``.
        padding_value (float, optional): the value used to pad when
            ``padding="constant"``. Defaults to ``0``.
        time_dim (int, optional): the dimension of ``tensor`` that holds the
            time axis. Must be negative. Defaults to ``-1``.
        done_mask (torch.Tensor, optional): an optional boolean mask flagging,
            for each sliding window, which of its ``N`` positions reach across
            a trajectory boundary (and must therefore be padded). Its shape is
            ``(*batch, time, N)`` where ``time`` matches the size of ``tensor``
            along ``time_dim``. When ``None`` (default), the input is treated as
            a single trajectory and only the leading ``N - 1`` start-of-sequence
            frames are padded. :class:`~torchrl.envs.transforms.CatFrames`
            builds this mask from the environment ``done`` signal.

    Returns:
        torch.Tensor: a tensor identical to ``tensor`` except that its size
        along ``dim`` is multiplied by ``N`` (the concatenated window) and its
        dtype / device are preserved.

    Examples:
        >>> import torch
        >>> from torchrl.envs.transforms.functional import cat_frames
        >>> # a single trajectory of 4 frames, each a length-2 feature vector,
        >>> # stacked over a window of N=3 along the feature dim (-1).
        >>> frames = torch.arange(8.0).view(4, 2)
        >>> frames
        tensor([[0., 1.],
                [2., 3.],
                [4., 5.],
                [6., 7.]])
        >>> out = cat_frames(frames, N=3, dim=-1, time_dim=-2, padding="constant")
        >>> out.shape
        torch.Size([4, 6])
        >>> out
        tensor([[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 1., 2., 3.],
                [0., 1., 2., 3., 4., 5.],
                [2., 3., 4., 5., 6., 7.]])

    .. note:: This functional covers the **offline** (contiguous trajectory
        slice) windowing used by
        :class:`~torchrl.envs.transforms.CatFrames`. The transform's
        **online** path (per-:meth:`~torchrl.envs.EnvBase.step` buffer
        accumulation) is inherently stateful and is not expressed as a pure
        function.

    .. seealso:: :class:`~torchrl.envs.transforms.CatFrames`.
    """
    if padding not in ("same", "constant"):
        raise ValueError(
            f"padding must be one of 'same' or 'constant', got {padding!r}."
        )
    if time_dim >= 0:
        raise ValueError("time_dim must be < 0.")
    if done_mask is None:
        # Treat the input as a single contiguous trajectory: only the leading
        # N - 1 start-of-sequence windows have padded (out-of-trajectory)
        # positions. Window position ``j`` of output step ``t`` reads source
        # step ``t - (N - 1) + j``; it is padded when that index is < 0.
        time_pos = tensor.ndim + time_dim
        time_len = tensor.shape[time_pos]
        steps = torch.arange(time_len, device=tensor.device)
        positions = torch.arange(N, device=tensor.device)
        # shape (time_len, N): True where the source index is before the start
        mask = positions[None, :] < (N - 1 - steps)[:, None]
        # broadcast over the leading batch dims (everything before time) and
        # add a trailing singleton "done feature" dim so the mask matches the
        # ``(*batch, time, 1, N)`` layout produced from a per-step done signal.
        shape = [1] * (time_pos + 1) + [1, N]
        shape[time_pos] = time_len
        done_mask = (
            mask.unsqueeze(-2)
            .reshape(shape)
            .expand(*tensor.shape[: time_pos + 1], 1, N)
        )
    data = _cat_frames_windows(
        tensor,
        N,
        dim,
        padding=padding,
        padding_value=padding_value,
        time_dim=time_dim,
        done_mask=done_mask,
    )
    return data.flatten(data.ndim + dim - 1, data.ndim + dim)
