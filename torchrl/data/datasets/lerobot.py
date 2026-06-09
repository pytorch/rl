# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LeRobot dataset adapter mapping to the canonical VLA TensorDict schema."""
from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from tensordict import NonTensorStack, TensorDict, TensorDictBase
from tensordict.utils import NestedKey

from torchrl._utils import logger as torchrl_logger
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

if TYPE_CHECKING:
    from torchrl.envs import Transform
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.samplers import Sampler, SliceSampler
from torchrl.data.replay_buffers.storages import _collate_id, TensorStorage
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer
from torchrl.data.video import _has_torchcodec, VideoClipRef

_has_lerobot = importlib.util.find_spec("lerobot") is not None

__all__ = ["LeRobotExperienceReplay", "lerobot_columns_to_tensordict"]

#: Default mapping from LeRobot's dotted column names to canonical VLA keys.
#: ``observation.images.<camera>`` columns are mapped automatically to
#: ``("observation", "image", <camera>)``; any other dotted name is split into a
#: nested key.
_DEFAULT_KEY_MAP: dict[str, NestedKey] = {
    "action": "action",
    "observation.state": ("observation", "state"),
    "episode_index": "episode",
    "frame_index": "frame",
    "task": "language_instruction",
    "next.reward": ("next", "reward"),
    "next.done": ("next", "done"),
}

_IMAGE_PREFIX = "observation.images."


def _map_lerobot_key(name: str, key_map: dict[str, NestedKey]) -> NestedKey:
    if name in key_map:
        return key_map[name]
    if name.startswith(_IMAGE_PREFIX):
        return ("observation", "image", name[len(_IMAGE_PREFIX) :])
    return tuple(name.split(".")) if "." in name else name


def lerobot_columns_to_tensordict(
    columns: dict[str, Any], *, key_map: dict[str, NestedKey] | None = None
) -> TensorDict:
    """Convert a LeRobot-style columnar dict into a canonical VLA TensorDict.

    LeRobot stores per-frame data under dotted column names
    (``observation.state``, ``observation.images.<camera>``, ``action``,
    ``episode_index``, ``task``, ...). This builds a flat ``[N]`` TensorDict
    using the canonical VLA key layout: proprioceptive state and images under
    ``observation``, the per-frame language instruction and the action at the
    root, and ``episode`` for trajectory boundaries (see
    :func:`~torchrl.data.vla.validate_vla_tensordict`).

    Args:
        columns (dict): mapping from LeRobot column name to a tensor (numeric
            columns), a list of strings (e.g. the ``task`` instruction), or a
            :class:`~torchrl.data.VideoClipRef` (lazy video frames, decoded on
            sampling by :class:`~torchrl.envs.transforms.DecodeVideoTransform`).

    Keyword Args:
        key_map (dict, optional): overrides/extends :data:`_DEFAULT_KEY_MAP`,
            mapping a source column name to a target :class:`~tensordict.utils.NestedKey`.

    Returns:
        a flat ``[N]`` :class:`~tensordict.TensorDict`.

    Examples:
        >>> import torch
        >>> from torchrl.data.datasets.lerobot import lerobot_columns_to_tensordict
        >>> columns = {
        ...     "observation.state": torch.zeros(4, 7),
        ...     "observation.images.top": torch.zeros(4, 3, 8, 8, dtype=torch.uint8),
        ...     "action": torch.zeros(4, 7),
        ...     "episode_index": torch.tensor([0, 0, 1, 1]),
        ...     "task": ["pick", "pick", "place", "place"],
        ... }
        >>> td = lerobot_columns_to_tensordict(columns)
        >>> td["observation", "state"].shape
        torch.Size([4, 7])
        >>> td["observation", "image", "top"].shape
        torch.Size([4, 3, 8, 8])
        >>> td.get("language_instruction").tolist()
        ['pick', 'pick', 'place', 'place']
    """
    key_map = {**_DEFAULT_KEY_MAP, **(key_map or {})}
    n = None
    for value in columns.values():
        if isinstance(value, (torch.Tensor, VideoClipRef)):
            n = value.shape[0]
            break
        if isinstance(value, (list, tuple)):
            n = len(value)
            break
    if n is None:
        raise ValueError("Could not infer the number of frames from `columns`.")

    out = TensorDict({}, batch_size=[n])
    for name, value in columns.items():
        target = _map_lerobot_key(name, key_map)
        if isinstance(value, (torch.Tensor, VideoClipRef)):
            out.set(target, value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, str) for v in value
        ):
            out.set(target, NonTensorStack.from_list(list(value)))
        elif isinstance(value, (list, tuple)) and any(
            isinstance(v, str) for v in value
        ):
            raise ValueError(
                f"Column {name!r} mixes strings and non-strings; expected a "
                "homogeneous column."
            )
        else:
            out.set(target, torch.as_tensor(value))
    return out


def _video_ref_keys(data: TensorDictBase) -> list[NestedKey]:
    """Return the keys of every :class:`~torchrl.data.VideoClipRef` leaf in ``data``.

    Walks nested tensordicts but does not descend into the references themselves,
    so the keys can be handed to
    :class:`~torchrl.envs.transforms.DecodeVideoTransform`.
    """
    keys: list[NestedKey] = []

    def _walk(td: TensorDictBase, prefix: tuple) -> None:
        for key in td.keys():
            value = td.get(key)
            nested = (*prefix, key)
            if isinstance(value, VideoClipRef):
                keys.append(nested[0] if len(nested) == 1 else nested)
            elif isinstance(value, TensorDictBase):
                _walk(value, nested)

    _walk(data, ())
    return keys


class LeRobotExperienceReplay(BaseDatasetExperienceReplay):
    """Experience replay over a `LeRobot <https://github.com/huggingface/lerobot>`_ dataset.

    LeRobot is the de-facto open format for robot-learning datasets (Parquet for
    state/action + MP4 for video), hosting many community datasets and the data
    used to train SmolVLA / pi0 / ACT. This adapter maps a LeRobot dataset into
    the canonical VLA TensorDict schema and serves it as a TorchRL replay buffer
    with trajectory-aware slice sampling.

    There are three ways to build it:

    - ``LeRobotExperienceReplay(repo_id, download=True)`` downloads and converts
      a dataset via the optional ``lerobot`` package (raises a helpful error if
      ``lerobot`` is not installed);
    - ``LeRobotExperienceReplay(repo_id, root=..., download=False)`` loads a
      previously-converted memory-mapped copy from disk;
    - :meth:`from_columns` builds directly from an in-memory LeRobot-style
      columnar dict (no download), which is also the path used in tests.

    Args:
        repo_id (str): the Hugging Face dataset repo id (e.g.
            ``"lerobot/aloha_sim_insertion_human"``).

    Keyword Args:
        root (str or Path, optional): local cache root. Defaults to the TorchRL
            LeRobot cache directory.
        download (bool): whether to download+convert the dataset if it is not
            already cached. Defaults to ``True``.
        batch_size (int, optional): the batch size for sampling.
        num_slices (int, optional): number of trajectory slices per batch
            (exclusive with ``slice_len``).
        slice_len (int, optional): length of each trajectory slice.
        sampler (Sampler, optional): a custom sampler. Defaults to a
            :class:`~torchrl.data.SliceSampler` over the ``episode`` key when
            ``num_slices``/``slice_len`` is given.
        writer (Writer, optional): a custom writer.
        transform (Transform, optional): a transform applied on sampling.
        key_map (dict, optional): overrides the default LeRobot-to-canonical key
            mapping (see :func:`lerobot_columns_to_tensordict`).
        decode_video (bool): if ``True`` (default) and the dataset carries lazy
            :class:`~torchrl.data.VideoClipRef` video columns, a
            :class:`~torchrl.envs.transforms.DecodeVideoTransform` is appended so
            that ``sample()`` returns decoded frames (requires ``torchcodec``).
            Set to ``False`` to keep the raw references and decode them yourself.
        strict_length (bool): passed to the slice sampler. Defaults to ``True``.
        collate_fn (Callable, optional): merges samples; defaults to the
            identity collation used by offline datasets.
        pin_memory (bool): whether to pin memory on sampling. Defaults to ``False``.
        prefetch (int, optional): number of batches to prefetch with a background
            thread.

    .. note::
        Sampled batches are *flat* ``[num_slices * slice_len]`` like any
        :class:`~torchrl.data.SliceSampler` output; reshape to
        ``[num_slices, slice_len, ...]`` before applying
        :class:`~torchrl.envs.transforms.ActionChunkTransform`.

    .. note::
        MP4 video columns are loaded lazily as :class:`~torchrl.data.VideoClipRef`
        leaves -- no frames are materialized in storage. With ``decode_video=True``
        (the default) they are decoded on ``sample()`` via
        :class:`~torchrl.envs.transforms.DecodeVideoTransform` (requires
        ``torchcodec``).

    .. warning::
        The ``download=True`` path is written against the documented LeRobot
        API and is **best-effort / not exercised in CI** (``lerobot`` is an
        optional dependency). The ``LeRobotDataset`` import path and the
        instruction column layout (string ``task`` vs integer ``task_index`` +
        a tasks lookup) vary across ``lerobot`` versions and may need
        adjusting. For fully reproducible behavior, build offline via
        :meth:`from_columns`.

    Examples:
        >>> import torch
        >>> from torchrl.data.datasets import LeRobotExperienceReplay
        >>> columns = {
        ...     "observation.state": torch.zeros(8, 7),
        ...     "action": torch.zeros(8, 7),
        ...     "episode_index": torch.arange(2).repeat_interleave(4),
        ...     "task": ["pick"] * 8,
        ... }
        >>> rb = LeRobotExperienceReplay.from_columns(
        ...     columns, slice_len=4, batch_size=8
        ... )
        >>> sample = rb.sample()
        >>> sample["action"].shape
        torch.Size([8, 7])

    .. seealso:: :class:`~torchrl.data.datasets.OpenXExperienceReplay` for the
        Open X-Embodiment equivalent.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        download: bool = True,
        batch_size: int | None = None,
        num_slices: int | None = None,
        slice_len: int | None = None,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        transform: Transform | None = None,
        key_map: dict[str, NestedKey] | None = None,
        decode_video: bool = True,
        strict_length: bool = True,
        pin_memory: bool = False,
        prefetch: int | None = None,
        _data: TensorDictBase | None = None,
    ) -> None:
        if (num_slices is not None) and (slice_len is not None):
            raise ValueError("num_slices or slice_len can be passed, but not both.")
        self.repo_id = repo_id
        self.key_map = key_map
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.strict_length = strict_length
        if root is None:
            root = _get_root_dir("lerobot")
        self.root = Path(root)
        # Only touch disk when a download or memmap-load will actually happen;
        # the in-memory ``from_columns`` path (``_data`` set) stays disk-free.
        if _data is None:
            os.makedirs(self.root, exist_ok=True)

        if _data is not None:
            data = _data
        elif download and not self._is_downloaded():
            data = self._download_and_preproc()
        elif self._is_downloaded():
            data = TensorDict.load_memmap(self.data_path)
        else:
            raise RuntimeError(
                f"Dataset {repo_id!r} not found at {self.data_path}. Pass "
                "download=True to fetch it (requires the `lerobot` package)."
            )
        storage = TensorStorage(data)

        # Decode lazy VideoClipRef leaves on the sample path: the storage keeps
        # only the lightweight references (no materialized frames), and frames are
        # decoded when the buffer is sampled. Disable with ``decode_video=False``
        # to keep the raw references.
        video_keys = _video_ref_keys(data)
        if decode_video and video_keys:
            if _has_torchcodec:
                from torchrl.envs.transforms import Compose, DecodeVideoTransform

                decode = DecodeVideoTransform(in_keys=video_keys)
                transform = decode if transform is None else Compose(decode, transform)
            else:
                torchrl_logger.warning(
                    "LeRobotExperienceReplay: video-frame references are present "
                    "but torchcodec is not installed, so frames will not be decoded "
                    "on sampling. Install torchcodec (`pip install "
                    "'torchcodec>=0.10.0'`) or pass decode_video=False to keep the "
                    "raw references and silence this warning."
                )

        if sampler is None and (num_slices is not None or slice_len is not None):
            sampler = SliceSampler(
                num_slices=num_slices,
                slice_len=slice_len,
                traj_key="episode",
                strict_length=strict_length,
            )
        if writer is None:
            writer = ImmutableDatasetWriter()
        if collate_fn is None:
            collate_fn = _collate_id
        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            batch_size=batch_size,
            transform=transform,
            pin_memory=pin_memory,
            prefetch=prefetch,
        )

    @classmethod
    def from_columns(
        cls,
        columns: dict[str, Any],
        *,
        repo_id: str = "local",
        key_map: dict[str, NestedKey] | None = None,
        **kwargs,
    ) -> LeRobotExperienceReplay:
        """Build directly from an in-memory LeRobot-style columnar dict.

        Converts ``columns`` with :func:`lerobot_columns_to_tensordict` and
        wraps the result in an in-memory storage -- no download or ``lerobot``
        install required.
        """
        data = lerobot_columns_to_tensordict(columns, key_map=key_map)
        return cls(repo_id, download=False, _data=data, key_map=key_map, **kwargs)

    @property
    def data_path_root(self) -> Path:
        return self.root / self.repo_id.replace("/", "_")

    @property
    def data_path(self) -> Path:
        return self.data_path_root

    def _is_downloaded(self) -> bool:
        return self.data_path.exists()

    def _download_and_preproc(self) -> TensorDictBase:
        if not _has_lerobot:
            raise ImportError(
                "The `lerobot` package is required to download LeRobot datasets. "
                "Install it with `pip install lerobot`, or build the dataset "
                "offline (e.g. via LeRobotExperienceReplay.from_columns) and load "
                "it with download=False."
            )
        # Lazy import of the optional `lerobot` dependency. NB: this import path
        # and the dataset layout are written against the documented LeRobot API
        # and are not exercised in CI -- see the class docstring warning.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(self.repo_id)
        columns = self._extract_columns(dataset)
        data = lerobot_columns_to_tensordict(columns, key_map=self.key_map)
        os.makedirs(self.data_path, exist_ok=True)
        data.memmap_(self.data_path)
        return data

    @staticmethod
    def _extract_columns(dataset) -> dict[str, Any]:
        """Read a LeRobotDataset's frames into a columnar dict.

        .. note::
            Written against the documented LeRobot API and **not exercised in
            CI**. MP4 video columns are mapped to lazy
            :class:`~torchrl.data.VideoClipRef` leaves (best-effort, spanning the
            per-episode video files); if the video layout cannot be resolved the
            column is skipped with a warning, as are any other non-tensor columns.
            The per-frame instruction is taken from a string ``task`` column, or
            resolved from an integer ``task_index`` joined against
            ``dataset.meta.tasks`` (the common LeRobot layout).
        """
        hf_dataset = dataset.hf_dataset.with_format("torch")
        video_keys = set(
            getattr(getattr(dataset, "meta", None), "video_keys", None) or []
        )
        columns: dict[str, Any] = {}
        for name in hf_dataset.column_names:
            if name in video_keys:
                ref = LeRobotExperienceReplay._build_video_ref(dataset, name)
                if ref is not None:
                    columns[name] = ref
                    continue
                # could not resolve the video layout: fall through to the skip
            column = hf_dataset[name]
            if isinstance(column, list) and column and isinstance(column[0], str):
                columns[name] = column
                continue
            try:
                columns[name] = torch.as_tensor(column)
            except (TypeError, ValueError, RuntimeError):
                torchrl_logger.warning(
                    f"LeRobotExperienceReplay: skipping column {name!r} which "
                    "could not be converted to a tensor or a VideoClipRef."
                )
        if "task" not in columns and "task_index" in columns:
            tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
            if tasks is not None:
                columns["task"] = [
                    str(tasks[int(i)]) for i in columns["task_index"].tolist()
                ]
        return columns

    @staticmethod
    def _build_video_ref(dataset, key: str) -> VideoClipRef | None:
        """Best-effort lazy :class:`~torchrl.data.VideoClipRef` for a video column.

        Spans the per-episode MP4 files of ``key`` in episode order so that frame
        ``j`` of the reference lines up with row ``j`` of the (episode-major)
        dataset. Per-episode frame counts are derived from ``episode_index`` (no
        metadata read). Returns ``None`` on any failure so the caller falls back to
        skipping the column. **Not exercised in CI** (see :meth:`_extract_columns`).
        """
        try:
            meta = dataset.meta
            root = Path(dataset.root)
            ep_index = torch.as_tensor(dataset.hf_dataset["episode_index"])
            ep_order = list(dict.fromkeys(int(e) for e in ep_index.tolist()))
            lengths = [int((ep_index == ep).sum()) for ep in ep_order]
            paths = [str(root / meta.get_video_file_path(ep, key)) for ep in ep_order]
            return VideoClipRef.from_files(paths, num_frames_per_file=lengths)
        except Exception as err:
            torchrl_logger.warning(
                f"LeRobotExperienceReplay: could not build a VideoClipRef for video "
                f"column {key!r} ({type(err).__name__}: {err}); skipping it."
            )
            return None
