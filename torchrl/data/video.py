# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Lazy, picklable references to frames inside encoded videos.

This module provides :class:`VideoClipRef`, a small :class:`~tensordict.TensorClass`
that stores *where* frames live (a video path/URI plus per-frame indices) without
materializing or decoding them. Indexing a reference is cheap and stays lazy;
decoding to a dense ``uint8`` tensor happens explicitly via
:meth:`VideoClipRef.decode` (or implicitly, see ``auto_decode``) using torchcodec.

It is dataset-agnostic on purpose: anything whose rows carry ``(video file, frame
position)`` can populate a :class:`VideoClipRef`, and decoding never needs to know
which dataset it came from.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from collections import defaultdict, OrderedDict
from typing import Any

import torch
from tensordict import TensorClass

from torchrl._utils import logger as torchrl_logger

__all__ = [
    "VideoClipRef",
    "clear_video_decoder_cache",
    "set_video_decoder_cache_size",
]

_has_torchcodec = importlib.util.find_spec("torchcodec") is not None

_TORCHCODEC_ERROR = (
    "This feature requires torchcodec >= 0.10.0. When running TorchRL from this "
    "repository with uv, use `uv run --extra video <command>` so torchcodec is "
    "installed in the command environment. Otherwise install it with "
    "`pip install 'torchcodec>=0.10.0'`."
)

# --- Per-process decoder cache ------------------------------------------------
# A torchcodec ``VideoDecoder`` holds C++ state + an open file descriptor: it is
# neither picklable nor safe to share across processes. We therefore NEVER store a
# decoder on a ``VideoClipRef`` (the reference carries only the address). Decoders
# are opened lazily and cached at module level, so the cache is naturally
# per-process: it is rebuilt independently in every collector / replay-buffer
# prefetch / DataLoader worker, and is never part of any pickled state.
_DECODER_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_DECODER_CACHE_LOCK = threading.Lock()
_DECODER_CACHE_MAXSIZE = 8


def set_video_decoder_cache_size(maxsize: int) -> None:
    """Sets the maximum number of open torchcodec decoders cached per process.

    The cache is keyed by ``(source, stream, device)``; least-recently-used
    decoders are evicted (and closed) once the limit is exceeded.

    Args:
        maxsize (int): the maximum number of decoders to keep open per process.
    """
    global _DECODER_CACHE_MAXSIZE
    _DECODER_CACHE_MAXSIZE = int(maxsize)
    with _DECODER_CACHE_LOCK:
        while len(_DECODER_CACHE) > _DECODER_CACHE_MAXSIZE:
            _DECODER_CACHE.popitem(last=False)


def clear_video_decoder_cache() -> None:
    """Closes and clears all cached torchcodec decoders in the current process."""
    with _DECODER_CACHE_LOCK:
        _DECODER_CACHE.clear()


def _get_decoder(source: Any, stream: int | None, device: Any):
    key = (source, stream, str(device) if device is not None else None)
    with _DECODER_CACHE_LOCK:
        decoder = _DECODER_CACHE.get(key)
        if decoder is not None:
            _DECODER_CACHE.move_to_end(key)
            return decoder
    if not _has_torchcodec:
        raise ModuleNotFoundError(_TORCHCODEC_ERROR)
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as err:  # pragma: no cover - import-time environment issue
        raise ImportError(_TORCHCODEC_ERROR) from err
    kwargs: dict[str, Any] = {}
    if stream is not None:
        kwargs["stream_index"] = int(stream)
    if device is not None:
        kwargs["device"] = str(device)
    decoder = VideoDecoder(source, **kwargs)
    with _DECODER_CACHE_LOCK:
        _DECODER_CACHE[key] = decoder
        _DECODER_CACHE.move_to_end(key)
        while len(_DECODER_CACHE) > _DECODER_CACHE_MAXSIZE:
            _DECODER_CACHE.popitem(last=False)
    return decoder


def _num_frames(decoder) -> int:
    num = getattr(decoder.metadata, "num_frames", None)
    if num is None:
        num = len(decoder)
    return int(num)


def _flatten(value: Any) -> list:
    """Flattens nested non-tensor leaves (lists / NonTensorStack / LinkedList)."""
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, os.PathLike):
        return [os.fspath(value)]
    if hasattr(value, "tolist") and not torch.is_tensor(value):
        # NonTensorStack and similar expose a python view via tolist()
        return _flatten(value.tolist())
    if torch.is_tensor(value):
        return value.reshape(-1).tolist()
    try:
        iterator = iter(value)
    except TypeError:
        return [value]
    out: list = []
    for element in iterator:
        out.extend(_flatten(element))
    return out


def _first(value: Any, default: Any = None) -> Any:
    flat = _flatten(value)
    for element in flat:
        return element
    return default


def _is_contiguous_run(indices: list[int]) -> bool:
    return len(indices) > 0 and indices == list(
        range(indices[0], indices[0] + len(indices))
    )


# Whether GPU decoding has been found unsupported by the installed torchcodec
# build in this process (set the first time a CUDA decode is attempted and fails).
_CUDA_DECODE_DISABLED = False


def _is_unsupported_device_error(err: Exception) -> bool:
    message = str(err).lower()
    return "unsupported device" in message or "deviceinterface" in message


def _frames_for_indices(decoder, indices: list[int]) -> torch.Tensor:
    if _is_contiguous_run(indices):
        return decoder.get_frames_in_range(start=indices[0], stop=indices[-1] + 1).data
    return decoder.get_frames_at(indices=indices).data


def _decode_group(
    source: Any, stream: int | None, indices: list[int], decode_device: Any
) -> torch.Tensor:
    """Decodes frames for a single source, with a CPU fallback.

    Falls back to CPU decoding when the torchcodec build cannot decode on the
    requested CUDA device (NVDEC); the caller moves the result to the requested
    output device afterwards.
    """
    global _CUDA_DECODE_DISABLED
    use_device = decode_device if not _CUDA_DECODE_DISABLED else None
    try:
        return _frames_for_indices(_get_decoder(source, stream, use_device), indices)
    except RuntimeError as err:
        if use_device is not None and _is_unsupported_device_error(err):
            _CUDA_DECODE_DISABLED = True
            torchrl_logger.warning(
                "torchcodec cannot decode on the requested CUDA device; falling "
                "back to CPU decoding and moving frames to the device. Install a "
                "CUDA-enabled torchcodec build to use NVDEC."
            )
            return _frames_for_indices(_get_decoder(source, stream, None), indices)
        raise


def _bin_frame_index(
    num_frames: int, num_bins: int, frames_per_bin: int | None = None
) -> torch.Tensor:
    """Builds frame positions for ``num_bins`` non-overlapping temporal bins.

    Bins partition ``[0, num_frames)`` via ``round(linspace(0, num_frames,
    num_bins + 1))`` edges. With ``frames_per_bin=None`` each bin contributes its
    center frame (shape ``[num_bins]`` -- a subsample). With ``frames_per_bin=k``
    each bin contributes ``k`` frames spanning the bin (shape ``[num_bins, k]`` -- a
    dense stack; frames are dropped or repeated as needed to stay rectangular).
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
    edges = torch.linspace(0, num_frames, num_bins + 1).round().long()
    last = num_frames - 1
    lo = edges[:-1].clamp(0, last)
    hi = torch.maximum((edges[1:] - 1).clamp(0, last), lo)
    if frames_per_bin is None:
        return ((lo + hi).to(torch.float) / 2).round().long()
    k = int(frames_per_bin)
    if k < 1:
        raise ValueError(f"frames_per_bin must be >= 1, got {k}.")
    fractions = torch.linspace(0, 1, k)
    span = (hi - lo).to(torch.float)
    return (
        (lo[:, None].to(torch.float) + fractions[None, :] * span[:, None])
        .round()
        .long()
    )


class VideoClipRef(TensorClass["nocast"]):
    """A lazy, picklable reference to frames inside an encoded video.

    A ``VideoClipRef`` stores only the *address* of frames -- a video ``source``
    (path / URI / file id) plus a ``frame_index`` tensor -- never an open decoder
    and never the decoded pixels. Its batch dimensions are the frames, so indexing
    behaves like a tensor of frames but stays lazy::

        video[50:100]   # -> VideoClipRef with 50 frames, no decoding happened

    Decoding to a dense ``uint8`` tensor is explicit, batched and seek-friendly via
    :meth:`decode` (grouping by source, using ranged reads for contiguous indices),
    which makes it compose naturally with :class:`~torchrl.data.SliceSampler` (a
    contiguous window of steps becomes a single ranged decode). Use the companion
    :class:`~torchrl.envs.transforms.DecodeVideoTransform` to decode on the
    replay-buffer sample path.

    To align a video onto a lower-rate signal (e.g. proprioceptive steps), use
    :meth:`rebin` (subsample to one frame per bin, or a dense non-overlapping
    per-bin stack) or :meth:`from_timestamps` (time-based alignment).

    Args:
        source (str): the video path / URI / file id. Required.
        frame_index (torch.Tensor, optional): a ``long`` tensor of frame positions
            whose shape becomes the batch size. If omitted, it defaults to *every*
            frame of ``source`` (``arange(num_frames)``) and the frame count is read
            from the file metadata once at construction (this opens the file and
            requires torchcodec). Pass it explicitly to reference a subset -- e.g.
            one episode of a multi-episode / "bucketed" file -- with no metadata read.

    Keyword Args:
        stream (int, optional): the video stream index to decode from. ``None``
            (default) lets torchcodec pick the best video stream.
        auto_decode (bool, optional): if ``True``, indexing the reference
            (``ref[...]``) returns decoded frames directly instead of a narrowed
            reference. Intended for standalone / interactive use; keep it ``False``
            for references stored in a replay buffer and rely on
            :class:`~torchrl.envs.transforms.DecodeVideoTransform` instead.
            Defaults to ``False``.
        out_device (torch.device or str, optional): default output device for
            decoded frames. A CUDA device uses GPU (NVDEC) decoding when the
            torchcodec build supports it, otherwise frames are decoded on CPU and
            moved to the device. Defaults to ``None`` (CPU).
        out_dtype (torch.dtype, optional): default dtype for decoded frames.
            Defaults to ``None`` (``uint8``).

    .. note:: A torchcodec ``VideoDecoder`` is not picklable nor process-safe.
        ``VideoClipRef`` is fully picklable because it stores no decoder; decoders
        are opened lazily and cached per process (see
        :func:`set_video_decoder_cache_size`).

    Examples:
        >>> import tempfile, os, torch
        >>> from torchcodec.encoders import VideoEncoder
        >>> from torchrl.data import VideoClipRef
        >>> frames = torch.arange(16, dtype=torch.uint8).reshape(16, 1, 1, 1)
        >>> frames = frames.expand(16, 3, 8, 8).contiguous()
        >>> path = os.path.join(tempfile.mkdtemp(), "clip.mp4")
        >>> VideoEncoder(frames=frames, frame_rate=10).to_file(path)
        >>> ref = VideoClipRef(path)        # every frame; batch size read from file
        >>> ref.batch_size
        torch.Size([16])
        >>> clip = ref[4:8]            # lazy, no decoding
        >>> clip.batch_size
        torch.Size([4])
        >>> decoded = clip.decode()    # uint8 [4, 3, 8, 8]
        >>> decoded.shape
        torch.Size([4, 3, 8, 8])

    .. seealso:: :class:`~torchrl.envs.transforms.DecodeVideoTransform`.
    """

    source: str
    frame_index: torch.Tensor | None = None
    stream: int | None = None
    auto_decode: bool = False
    out_device: Any = None
    out_dtype: Any = None

    def __post_init__(self):
        # Runs only on direct construction (``VideoClipRef(...)``), not on the
        # internal ``_from_tensordict`` path used by indexing/stacking -- and even
        # if it did run there, every branch below is a no-op once ``frame_index`` is
        # a long tensor with a matching ``batch_size``.
        frame_index = self.frame_index
        if frame_index is None:
            # No frame_index given: address every frame, reading the count from the
            # file metadata once. This is the ``VideoClipRef(path)`` ergonomic.
            frame_index = torch.arange(
                _num_frames(_get_decoder(self.source, self.stream, None))
            )
        elif not torch.is_tensor(frame_index):
            frame_index = torch.as_tensor(frame_index)
        if frame_index.dtype != torch.long:
            frame_index = frame_index.to(torch.long)
        if frame_index is not self.frame_index:
            self.frame_index = frame_index
        # tensorclass does not infer batch_size from a tensor field, so set it here
        # (the frames are the batch dimensions). A scalar (0-d) index stays scalar.
        if self.batch_size == torch.Size([]) and frame_index.ndim >= 1:
            self.batch_size = frame_index.shape

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike,
        *,
        stream: int | None = None,
        num_frames: int | None = None,
        frame_index: torch.Tensor | None = None,
        num_bins: int | None = None,
        frames_per_bin: int | None = None,
        auto_decode: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> VideoClipRef:
        """Builds a reference to (by default) every frame of a video file.

        Args:
            path (str or PathLike): the video path / URI / file id.

        Keyword Args:
            stream (int, optional): stream index to decode from. Defaults to
                ``None`` (best video stream).
            num_frames (int, optional): number of frames. If ``None`` and
                ``frame_index`` is also ``None``, the count is read from the file
                metadata (this opens the file once).
            frame_index (torch.Tensor, optional): explicit frame positions. If
                given, ``num_frames`` is ignored and no metadata read is performed.
                Mutually exclusive with ``num_bins``.
            num_bins (int, optional): resample the video onto this many
                non-overlapping temporal bins (see :meth:`rebin`). Defaults to
                ``None`` (every frame).
            frames_per_bin (int, optional): with ``num_bins``, the number of frames
                per bin (a dense ``[num_bins, frames_per_bin]`` stack). ``None``
                (default) takes a single center frame per bin (a ``[num_bins]``
                subsample). Requires ``num_bins``.
            auto_decode (bool, optional): see :class:`VideoClipRef`. Defaults to
                ``False``.
            device (torch.device or str, optional): default output device. A CUDA
                device uses GPU (NVDEC) decoding when supported, else decodes on CPU
                and moves the frames. Defaults to ``None``.
            dtype (torch.dtype, optional): default decode dtype. Defaults to
                ``None`` (``uint8``).

        Returns:
            VideoClipRef: a reference whose batch size is the number of frames (or
            ``[num_bins]`` / ``[num_bins, frames_per_bin]`` when binning).
        """
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        if num_bins is not None:
            if frame_index is not None:
                raise ValueError("`num_bins` and `frame_index` are mutually exclusive.")
            if num_frames is None:
                num_frames = _num_frames(_get_decoder(path, stream, None))
            frame_index = _bin_frame_index(
                int(num_frames), int(num_bins), frames_per_bin
            )
        elif frames_per_bin is not None:
            raise ValueError("`frames_per_bin` requires `num_bins`.")
        elif frame_index is None and num_frames is not None:
            frame_index = torch.arange(int(num_frames))
        # ``frame_index=None`` lets ``__post_init__`` read the frame count from the
        # file metadata; an explicit ``frame_index`` (or ``num_frames``) skips that.
        return cls(
            source=path,
            frame_index=frame_index,
            stream=stream,
            auto_decode=auto_decode,
            out_device=device,
            out_dtype=dtype,
        )

    @classmethod
    def from_timestamps(
        cls,
        path: str | os.PathLike,
        timestamps: torch.Tensor,
        *,
        stream: int | None = None,
        fps: float | None = None,
        auto_decode: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> VideoClipRef:
        """Builds a reference from timestamps (seconds), for fps-mismatched streams.

        Timestamps are converted to the nearest frame index using the stream's
        average fps (read from the file unless ``fps`` is given), so cameras that
        run at a different rate than the control loop can still be addressed.

        Args:
            path (str or PathLike): the video path / URI / file id.
            timestamps (torch.Tensor): timestamps in seconds.

        Keyword Args:
            stream (int, optional): stream index. Defaults to ``None``.
            fps (float, optional): frames-per-second to use for the conversion.
                Defaults to ``None`` (read from metadata).
            auto_decode (bool, optional): see :class:`VideoClipRef`.
            device (torch.device or str, optional): default decode device.
            dtype (torch.dtype, optional): default decode dtype.

        Returns:
            VideoClipRef: a reference addressing the nearest frames to ``timestamps``.
        """
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        decoder = _get_decoder(path, stream, None)
        if fps is None:
            fps = float(decoder.metadata.average_fps)
        num_frames = _num_frames(decoder)
        timestamps = torch.as_tensor(timestamps, dtype=torch.float)
        frame_index = (timestamps * fps).round().long().clamp_(0, num_frames - 1)
        return cls.from_file(
            path,
            stream=stream,
            frame_index=frame_index,
            auto_decode=auto_decode,
            device=device,
            dtype=dtype,
        )

    def rebin(self, num_bins: int, frames_per_bin: int | None = None) -> VideoClipRef:
        """Resamples the referenced frames onto ``num_bins`` non-overlapping bins.

        Bins partition the frames this reference currently addresses into
        ``num_bins`` contiguous, non-overlapping temporal bins, useful to align a
        video onto a lower-rate signal (e.g. proprioceptive steps).

        - ``frames_per_bin=None`` (default) keeps one **center** frame per bin: the
          returned reference has batch size ``[num_bins]`` and decodes to
          ``[num_bins, C, H, W]`` (a subsample / decimation).
        - ``frames_per_bin=k`` keeps ``k`` frames spanning each bin: batch size
          ``[num_bins, k]``, decoding to ``[num_bins, k, C, H, W]`` (a dense,
          non-overlapping stack; frames are dropped or repeated to stay rectangular).

        For *overlapping* (sliding-window) stacking, subsample first
        (``rebin(num_bins)``) and apply :class:`~torchrl.envs.transforms.CatFrames`
        to the decoded frames on the sample path.

        Args:
            num_bins (int): the number of temporal bins.
            frames_per_bin (int, optional): frames kept per bin. ``None`` (default)
                takes the single center frame.

        Returns:
            VideoClipRef: a new (lazy) reference over the binned frames.

        Examples:
            >>> ref = VideoClipRef(path)            # 100 frames  # doctest: +SKIP
            >>> ref.rebin(30).batch_size            # one frame per proprio step
            torch.Size([30])
            >>> ref.rebin(30, frames_per_bin=3).batch_size
            torch.Size([30, 3])
        """
        base = self.frame_index.reshape(-1)
        positions = _bin_frame_index(int(base.numel()), int(num_bins), frames_per_bin)
        return type(self)(
            source=_first(self.source),
            frame_index=base[positions],
            stream=_first(self.stream),
            auto_decode=bool(_first(self.auto_decode, False)),
            out_device=_first(self.out_device),
            out_dtype=_first(self.out_dtype),
        )

    def _source_list(self, n: int) -> list:
        sources = _flatten(self.source)
        if len(sources) == 1 and n > 1:
            sources = sources * n
        return sources

    def decode(self, *, device: Any = None, dtype: Any = None) -> torch.Tensor:
        """Decodes the referenced frames to a dense tensor.

        Frames are grouped by ``source`` and decoded with a per-process decoder
        cache; contiguous index runs use a single ranged read. The output keeps the
        reference's batch shape: a scalar reference yields ``[C, H, W]`` and a
        reference with batch size ``[*B]`` yields ``[*B, C, H, W]``.

        Keyword Args:
            device (torch.device or str, optional): output device for the decoded
                frames, overriding ``out_device``. A CUDA device uses GPU (NVDEC)
                decoding when the torchcodec build supports it, and otherwise decodes
                on CPU and moves the frames to the device.
            dtype (torch.dtype, optional): dtype for the decoded frames, overriding
                ``out_dtype``. Defaults to ``uint8``.

        Returns:
            torch.Tensor: the decoded frames (``uint8`` by default).
        """
        if not _has_torchcodec:
            raise ModuleNotFoundError(_TORCHCODEC_ERROR)
        if device is None:
            device = _first(self.out_device)
        if dtype is None:
            dtype = _first(self.out_dtype)
        stream = _first(self.stream)
        out_device = torch.device(device) if device is not None else None
        # Only attempt on-device (NVDEC) decoding for CUDA outputs; CPU outputs and
        # the fallback path decode on CPU and move afterwards.
        decode_device = (
            out_device if out_device is not None and out_device.type == "cuda" else None
        )

        frame_index = self.frame_index.reshape(-1).cpu().tolist()
        n = len(frame_index)
        sources = self._source_list(n)
        if len(sources) != n:
            raise RuntimeError(
                f"VideoClipRef.decode found {len(sources)} sources for {n} frame "
                "indices; source and frame_index must describe the same elements."
            )

        groups: dict[Any, list[tuple[int, int]]] = defaultdict(list)
        for position, (src, idx) in enumerate(zip(sources, frame_index)):
            groups[src].append((position, int(idx)))

        out: list[torch.Tensor | None] = [None] * n
        for src, items in groups.items():
            # Decode each distinct frame once, then scatter to every position that
            # referenced it (binning / sliding windows can repeat indices).
            unique = sorted({idx for _, idx in items})
            data = _decode_group(src, stream, unique, decode_device)
            offset_of = {idx: offset for offset, idx in enumerate(unique)}
            for position, idx in items:
                out[position] = data[offset_of[idx]]

        stacked = torch.stack(out)  # type: ignore[arg-type]
        if dtype is not None:
            stacked = stacked.to(dtype)
        if out_device is not None:
            stacked = stacked.to(out_device)
        frame_shape = stacked.shape[1:]
        return stacked.reshape(*self.batch_size, *frame_shape)

    @property
    def frames(self) -> torch.Tensor:
        """Decoded frames for this reference (shorthand for :meth:`decode`)."""
        return self.decode()

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if isinstance(out, VideoClipRef) and bool(_first(out.auto_decode, False)):
            return out.decode()
        return out
