# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tensordict import TensorDictBase
from tensordict.utils import NestedKey

from torchrl.data.video import _has_torchcodec, _TORCHCODEC_ERROR, VideoClipRef
from torchrl.envs.transforms._base import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance

__all__ = ["DecodeVideoTransform"]


class DecodeVideoTransform(Transform):
    """Decodes :class:`~torchrl.data.VideoClipRef` leaves to dense frame tensors.

    This is a forward / sample-path transform: it reads the lazy video references
    found at ``in_keys`` and writes the decoded ``uint8`` frames at ``out_keys``. It
    is meant to be appended to a :class:`~torchrl.data.ReplayBuffer` so that
    indexing the buffer stays cheap (no materialized frames) while ``rb.sample()``
    returns decoded frames aligned to the sampled steps. It is a read-side codec, so
    no inverse is defined.

    Decoding is delegated to :meth:`VideoClipRef.decode`, which groups the sampled
    references by source file and uses ranged reads for contiguous indices. This is
    what makes it compose with :class:`~torchrl.data.SliceSampler`: a contiguous
    window of sampled steps maps to consecutive frame indices and decodes as a
    single ranged read per source.

    Keyword Args:
        in_keys (sequence of NestedKey): the keys holding the
            :class:`~torchrl.data.VideoClipRef` leaves to decode.
        out_keys (sequence of NestedKey, optional): destination keys for the decoded
            frames. Defaults to ``in_keys`` (in-place replacement).
        device (torch.device or str, optional): device for the decoded frames. A
            CUDA device enables GPU (NVDEC) decoding. Defaults to ``None`` (uses the
            reference's ``out_device``, else CPU).
        dtype (torch.dtype, optional): dtype for the decoded frames. Defaults to
            ``None`` (uses the reference's ``out_dtype``, else ``uint8``).

    .. note:: This transform requires torchcodec. The lightweight
        :class:`~torchrl.data.VideoClipRef` leaves stored in the buffer are
        picklable and hold no open decoder; decoders are opened lazily and cached
        per worker process.

    Examples:
        >>> import tempfile, os, torch
        >>> from torchcodec.encoders import VideoEncoder
        >>> from tensordict import TensorDict
        >>> from torchrl.data import (
        ...     LazyTensorStorage, ReplayBuffer, SliceSampler, VideoClipRef)
        >>> from torchrl.envs.transforms import DecodeVideoTransform
        >>> frames = torch.arange(20, dtype=torch.uint8).reshape(20, 1, 1, 1)
        >>> frames = frames.expand(20, 3, 8, 8).contiguous()
        >>> path = os.path.join(tempfile.mkdtemp(), "clip.mp4")
        >>> VideoEncoder(frames=frames, frame_rate=10).to_file(path)
        >>> ref = VideoClipRef.from_file(path)             # 20 frames, lazy
        >>> data = TensorDict(
        ...     {"frame": ref, "episode": torch.zeros(20, dtype=torch.long)},
        ...     batch_size=[20],
        ... )
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(20),
        ...     sampler=SliceSampler(slice_len=4, traj_key="episode"),
        ...     batch_size=8,
        ...     transform=DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"]),
        ... )
        >>> _ = rb.extend(data)
        >>> sample = rb.sample()
        >>> sample["pixels"].shape          # decoded on sample
        torch.Size([8, 3, 8, 8])

    .. seealso:: :class:`~torchrl.data.VideoClipRef`.
    """

    invertible = False

    def __init__(
        self,
        *,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey] | None = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        if not _has_torchcodec:
            raise ModuleNotFoundError(_TORCHCODEC_ERROR)
        if in_keys is None:
            raise TypeError("DecodeVideoTransform requires `in_keys`.")
        if out_keys is None:
            out_keys = list(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.device = device
        self.dtype = dtype

    def _apply_transform(self, value: VideoClipRef):
        if not isinstance(value, VideoClipRef):
            raise TypeError(
                "DecodeVideoTransform expected a VideoClipRef leaf, got "
                f"{type(value).__name__}."
            )
        return value.decode(device=self.device, dtype=self.dtype)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset
