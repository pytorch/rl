# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LeRobot dataset adapter mapping to the canonical VLA TensorDict schema."""
from __future__ import annotations

import importlib.util
import json
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

_has_datasets = importlib.util.find_spec("datasets") is not None
_has_hf_hub = importlib.util.find_spec("huggingface_hub") is not None

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
        if isinstance(value, torch.Tensor):
            # TED convention: per-step signals under "next" (reward, done,
            # success, ...) carry a trailing singleton dim. Without it,
            # samplers that combine these flags with their own [batch, 1]
            # entries (e.g. SliceSampler's truncated flag) would broadcast.
            if (
                isinstance(target, tuple)
                and target
                and target[0] == "next"
                and value.ndim == 1
            ):
                value = value.unsqueeze(-1)
            out.set(target, value)
        elif isinstance(value, VideoClipRef):
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


class _LeRobotSnapshot:
    """Direct reader for the LeRobot on-disk dataset format (v2.x and v3.x).

    Parses the hub snapshot files (``meta/info.json``, the data parquets and
    the task/episode metadata) without importing the ``lerobot`` package:
    only ``huggingface_hub`` and ``datasets`` are needed, and the reader is
    insulated from the ``lerobot`` package's torch version pins and API
    changes. Validated against ``lerobot/pusht`` (format ``v3.0``).
    """

    def __init__(self, repo_id: str) -> None:
        from huggingface_hub import snapshot_download

        self.repo_id = repo_id
        self.root = Path(snapshot_download(repo_id, repo_type="dataset"))
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        version = str(self.info.get("codebase_version", "v3.0")).lstrip("v")
        self.major_version = int(version.split(".")[0])
        self.fps = float(self.info["fps"])
        self.video_keys = [
            key
            for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
        self.hf_dataset = self._load_frames()
        self.episodes = self._load_episodes()
        self.tasks = self._load_tasks()

    def _load_frames(self):
        from datasets import load_dataset

        files = sorted(str(p) for p in (self.root / "data").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No data parquet files found under {self.root / 'data'}."
            )
        dataset = load_dataset("parquet", data_files=files, split="train")
        if "index" in dataset.column_names:
            index = list(dataset["index"])
            if index != sorted(index):
                dataset = dataset.sort("index")
        return dataset

    def _load_episodes(self) -> list[dict] | None:
        # v3 stores per-episode metadata (lengths, video file/timestamp spans)
        # in meta/episodes/*.parquet; v2.x uses meta/episodes.jsonl.
        meta_dir = self.root / "meta"
        episodes_dir = meta_dir / "episodes"
        rows: list[dict] = []
        if episodes_dir.exists():
            import pyarrow.parquet as pq  # ships with `datasets`

            for file in sorted(episodes_dir.rglob("*.parquet")):
                rows.extend(pq.read_table(file).to_pylist())
        elif (meta_dir / "episodes.jsonl").exists():
            with open(meta_dir / "episodes.jsonl") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            return None
        rows.sort(key=lambda row: int(row["episode_index"]))
        return rows

    def _load_tasks(self) -> dict[int, str]:
        # v3: meta/tasks.parquet with the task string as the table index;
        # v2.x: meta/tasks.jsonl with {"task_index": ..., "task": ...} rows.
        meta_dir = self.root / "meta"
        if (meta_dir / "tasks.parquet").exists():
            import pyarrow.parquet as pq

            out: dict[int, str] = {}
            for row in pq.read_table(meta_dir / "tasks.parquet").to_pylist():
                index = row.get("task_index")
                text = row.get("task", row.get("__index_level_0__"))
                if index is not None and isinstance(text, str):
                    out[int(index)] = text
            return out
        if (meta_dir / "tasks.jsonl").exists():
            with open(meta_dir / "tasks.jsonl") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            return {int(row["task_index"]): str(row["task"]) for row in rows}
        return {}

    def video_segments(self, key: str) -> tuple[list[str], list[int]]:
        """Ordered video files spanning ``key``, with per-file frame counts.

        Episodes are stored back-to-back inside each video file (v3) or one
        per file (v2.x), in episode order -- so the episode-major frame ``j``
        of the dataset is frame ``j`` of the concatenated files.
        """
        if self.episodes is None:
            raise RuntimeError(
                f"Missing episode metadata under {self.root / 'meta'} "
                "(expected 'episodes/*.parquet' or 'episodes.jsonl')."
            )
        template = self.info["video_path"]
        paths: list[str] = []
        counts: list[int] = []
        drift_warned = False
        for row in self.episodes:
            length = int(row["length"])
            if self.major_version >= 3:
                path = str(
                    self.root
                    / template.format(
                        video_key=key,
                        chunk_index=int(row[f"videos/{key}/chunk_index"]),
                        file_index=int(row[f"videos/{key}/file_index"]),
                    )
                )
                start = row.get(f"videos/{key}/from_timestamp")
                same_file = bool(paths) and paths[-1] == path
                expected_start = counts[-1] / self.fps if same_file else 0.0
                if (
                    not drift_warned
                    and start is not None
                    and abs(float(start) - expected_start) > 0.5 / self.fps
                ):
                    torchrl_logger.warning(
                        f"LeRobotExperienceReplay: episode "
                        f"{row['episode_index']} of video column {key!r} starts "
                        f"at {float(start):.3f}s but {expected_start:.3f}s was "
                        "expected from the cumulative episode lengths; frame "
                        "alignment may be off."
                    )
                    drift_warned = True
                if same_file:
                    counts[-1] += length
                else:
                    paths.append(path)
                    counts.append(length)
            else:
                # v2.x: one video file per episode, chunked by episode index.
                episode = int(row["episode_index"])
                chunk = episode // int(self.info.get("chunks_size", 1000))
                paths.append(
                    str(
                        self.root
                        / template.format(
                            episode_chunk=chunk,
                            video_key=key,
                            episode_index=episode,
                        )
                    )
                )
                counts.append(length)
        return paths, counts


class LeRobotExperienceReplay(BaseDatasetExperienceReplay):
    """Experience replay over a `LeRobot <https://github.com/huggingface/lerobot>`_ dataset.

    LeRobot is the de-facto open format for robot-learning datasets (Parquet for
    state/action + MP4 for video), hosting many community datasets and the data
    used to train SmolVLA / pi0 / ACT. This adapter maps a LeRobot dataset into
    the canonical VLA TensorDict schema and serves it as a TorchRL replay buffer
    with trajectory-aware slice sampling.

    There are three ways to build it:

    - ``LeRobotExperienceReplay(repo_id, download=True)`` downloads the hub
      snapshot and reads the on-disk LeRobot format (v2.x and v3.x) directly --
      only the ``huggingface_hub`` and ``datasets`` packages are required
      (installed by the ``vla`` extra), not the ``lerobot`` package itself;
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
            :class:`~torchrl.data.SliceSampler` over the (key-mapped) episode
            key -- ``episode`` unless ``key_map`` remaps ``episode_index`` --
            when ``num_slices``/``slice_len`` is given.
        writer (Writer, optional): a custom writer.
        transform (Transform, optional): a transform applied on sampling.
        key_map (dict, optional): overrides the default LeRobot-to-canonical key
            mapping (see :func:`lerobot_columns_to_tensordict`).
        decode_video (bool): if ``True`` (default) and the dataset carries lazy
            :class:`~torchrl.data.VideoClipRef` video columns, a
            :class:`~torchrl.envs.transforms.DecodeVideoTransform` is appended so
            that ``sample()`` returns decoded frames (requires ``torchcodec``).
            Set to ``False`` to keep the raw references and decode them yourself.
        rehydrate (bool): if ``True``, sampled batches are made fully
            TED-compliant by re-hydrating ``("next", "observation", ...)``
            entries from the following row of each sampled slice
            (:class:`~torchrl.envs.transforms.NextStateReconstructor`
            instances are appended after the video decode and before
            ``transform``). Boundaries are detected from the episode id
            (required) plus the per-episode frame counter (``frame``) when
            present, so positions whose in-batch successor is not the true
            next step -- slice ends and splices between back-to-back slices --
            are filled with ``NaN`` for floating-point leaves and ``0`` for
            integer leaves (e.g. decoded ``uint8`` frames); mask the filled
            positions with the slice sampler's ``("next", "truncated")`` flag
            in the returned batch when consuming ``next``. Video references
            left undecoded (``decode_video=False`` or ``torchcodec`` not
            installed) are skipped with a warning. Defaults to ``False``.
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
        The ``download=True`` path reads the documented LeRobot on-disk format
        (validated against ``lerobot/pusht``, format ``v3.0``) but is **not
        exercised in CI** (``huggingface_hub``/``datasets`` are optional
        dependencies and CI does not download datasets). For fully
        reproducible behavior, build offline via :meth:`from_columns`.

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
        >>> # rehydrate=True re-hydrates ("next", "observation", ...) from the
        >>> # following row of each slice (slice ends are filled and flagged
        >>> # by ("next", "truncated"))
        >>> rb = LeRobotExperienceReplay.from_columns(
        ...     columns, slice_len=4, batch_size=8, rehydrate=True
        ... )
        >>> sample = rb.sample()
        >>> sample["next", "observation", "state"].shape
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
        rehydrate: bool = False,
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
            data = self._attach_video_refs(TensorDict.load_memmap(self.data_path))
        else:
            raise RuntimeError(
                f"Dataset {repo_id!r} not found at {self.data_path}. Pass "
                "download=True to fetch it (requires the `huggingface_hub` "
                "and `datasets` packages)."
            )
        storage = TensorStorage(data)

        key_map_merged = {**_DEFAULT_KEY_MAP, **(key_map or {})}
        episode_key = _map_lerobot_key("episode_index", key_map_merged)

        # Decode lazy VideoClipRef leaves on the sample path: the storage keeps
        # only the lightweight references (no materialized frames), and frames are
        # decoded when the buffer is sampled. Disable with ``decode_video=False``
        # to keep the raw references.
        transforms = []
        video_keys = _video_ref_keys(data)
        videos_decoded = False
        if decode_video and video_keys:
            if _has_torchcodec:
                from torchrl.envs.transforms import DecodeVideoTransform

                transforms.append(DecodeVideoTransform(in_keys=video_keys))
                videos_decoded = True
            else:
                torchrl_logger.warning(
                    "LeRobotExperienceReplay: video-frame references are present "
                    "but torchcodec is not installed, so frames will not be decoded "
                    "on sampling. Install torchcodec (`pip install "
                    "'torchcodec>=0.10.0'`) or pass decode_video=False to keep the "
                    "raw references and silence this warning."
                )

        internal_slice_sampler = sampler is None and (
            num_slices is not None or slice_len is not None
        )
        if rehydrate:
            transforms.extend(
                self._make_rehydrate_transforms(
                    data,
                    video_keys=video_keys,
                    videos_decoded=videos_decoded,
                    episode_key=episode_key,
                    key_map=key_map_merged,
                )
            )
        if transform is not None:
            transforms.append(transform)
        if len(transforms) > 1:
            from torchrl.envs.transforms import Compose

            transform = Compose(*transforms)
        elif transforms:
            transform = transforms[0]

        if internal_slice_sampler:
            sampler = SliceSampler(
                num_slices=num_slices,
                slice_len=slice_len,
                traj_key=episode_key,
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

    @staticmethod
    def _make_rehydrate_transforms(
        data: TensorDictBase,
        *,
        video_keys: list[NestedKey],
        videos_decoded: bool,
        episode_key: NestedKey,
        key_map: dict[str, NestedKey],
    ) -> list:
        """Build the :class:`~torchrl.envs.transforms.NextStateReconstructor` instances for ``rehydrate=True``.

        Observation leaves are grouped by fill value: ``NaN`` for
        floating-point leaves, ``0`` for integer ones (e.g. decoded ``uint8``
        frames, where ``NaN`` cannot be represented). Video references that
        stay undecoded cannot be shifted and are skipped with a warning.

        Boundary detection only uses markers that live in **storage** (the
        buffer transforms run before the sampler's info entries, such as
        ``("next", "truncated")``, are merged into the batch): the episode id,
        the per-episode frame counter (``frame_index`` mapped key, used as a
        ``step_count_key`` -- it deterministically rejects back-to-back slices
        of the same episode) and the dataset's ``("next", "done")`` flag when
        present.
        """
        from torchrl.envs.transforms import NextStateReconstructor

        observation = data.get("observation", None)
        if observation is None:
            torchrl_logger.warning(
                "LeRobotExperienceReplay: rehydrate=True but the dataset has "
                "no 'observation' entry; nothing to re-hydrate."
            )
            return []
        leaves = set(data.keys(include_nested=True, leaves_only=True))
        if episode_key not in leaves:
            raise ValueError(
                "rehydrate=True requires the episode entry "
                f"({episode_key!r}) to delimit trajectories, but it is not "
                "present in the dataset."
            )
        frame_key = _map_lerobot_key("frame_index", key_map)
        if frame_key not in leaves:
            torchrl_logger.warning(
                "LeRobotExperienceReplay: rehydrate=True without a per-episode "
                f"frame counter ({frame_key!r}): back-to-back slices of the "
                "same episode cannot be told apart from a contiguous slice, "
                "so a spliced position may receive the first row of the next "
                "slice instead of the boundary fill."
            )
            frame_key = None
        done_key = key_map.get("next.done", ("next", "done"))
        if done_key not in leaves:
            done_key = None
        ref_keys = {key if isinstance(key, tuple) else (key,) for key in video_keys}
        ref_keys = {key for key in ref_keys if key[0] == "observation"}
        float_keys: list[NestedKey] = []
        int_keys: list[NestedKey] = []
        for key in observation.keys(include_nested=True, leaves_only=True):
            key_t = ("observation", *(key if isinstance(key, tuple) else (key,)))
            if any(key_t[: len(ref)] == ref for ref in ref_keys):
                # a field inside a video reference; the reference itself is
                # handled below
                continue
            if observation.get(key).dtype.is_floating_point:
                float_keys.append(key_t)
            else:
                int_keys.append(key_t)
        for ref in sorted(ref_keys):
            if videos_decoded:
                # decoded on the sample path before this transform runs; the
                # decode dtype follows the reference's out_dtype (uint8 when
                # unset)
                out_dtype = getattr(data.get(ref), "out_dtype", None)
                if isinstance(out_dtype, torch.dtype) and out_dtype.is_floating_point:
                    float_keys.append(ref)
                else:
                    int_keys.append(ref)
            else:
                torchrl_logger.warning(
                    "LeRobotExperienceReplay: rehydrate=True cannot shift the "
                    f"video reference at {ref!r}, which stays undecoded on "
                    "the sample path (decode_video=False or torchcodec not "
                    "installed); skipping it."
                )
        out = []
        if float_keys:
            out.append(
                NextStateReconstructor(
                    float_keys,
                    traj_key=episode_key,
                    done_key=done_key,
                    step_count_key=frame_key,
                )
            )
        if int_keys:
            out.append(
                NextStateReconstructor(
                    int_keys,
                    traj_key=episode_key,
                    done_key=done_key,
                    step_count_key=frame_key,
                    fill_value=0,
                )
            )
        return out

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
        if not (_has_hf_hub and _has_datasets):
            raise ImportError(
                "Downloading LeRobot datasets requires the `huggingface_hub` and "
                "`datasets` packages (installed by `pip install 'torchrl[vla]'` "
                "or `pip install lerobot`). Alternatively build the dataset "
                "offline (e.g. via LeRobotExperienceReplay.from_columns) and load "
                "it with download=False."
            )
        # The on-disk LeRobot dataset format (v2.x / v3.x) is read directly
        # from the hub snapshot: the heavy `lerobot` package is not needed (and
        # its torch pins would often conflict with the installed torch).
        dataset = _LeRobotSnapshot(self.repo_id)
        columns = self._extract_columns(dataset)
        # Lazy video references do not survive a memmap round-trip (the nested
        # tensorclass identity is lost on load), so they are kept out of the
        # memmapped storage and rebuilt from a sidecar manifest on every load.
        manifest: dict[str, dict] = {}
        for key in dataset.video_keys:
            if columns.pop(key, None) is None:
                continue
            paths, counts = dataset.video_segments(key)
            manifest[key] = {"paths": paths, "num_frames_per_file": counts}
        data = lerobot_columns_to_tensordict(columns, key_map=self.key_map)
        os.makedirs(self.data_path, exist_ok=True)
        data.memmap_(self.data_path)
        if manifest:
            with open(self.data_path / "video_refs.json", "w") as f:
                json.dump(manifest, f)
        return self._attach_video_refs(data)

    def _attach_video_refs(self, data: TensorDictBase) -> TensorDictBase:
        """Rebuild the lazy video references from the sidecar manifest.

        ``video_refs.json`` points at the MP4 files of the hub snapshot cache;
        the references are rebuilt every time the dataset is loaded (they are
        excluded from the memmapped storage, where the tensorclass identity
        would not survive the round-trip).
        """
        manifest_path = self.data_path / "video_refs.json"
        if not manifest_path.exists():
            return data
        with open(manifest_path) as f:
            manifest = json.load(f)
        key_map = {**_DEFAULT_KEY_MAP, **(self.key_map or {})}
        data = data.unlock_()
        for name, spec in manifest.items():
            ref = VideoClipRef.from_files(
                spec["paths"], num_frames_per_file=spec["num_frames_per_file"]
            )
            data.set(_map_lerobot_key(name, key_map), ref)
        return data

    @staticmethod
    def _extract_columns(dataset: _LeRobotSnapshot) -> dict[str, Any]:
        """Read a :class:`_LeRobotSnapshot`'s frames into a columnar dict.

        MP4 video columns are mapped to lazy
        :class:`~torchrl.data.VideoClipRef` leaves spanning the dataset's video
        files in episode order; if the video layout cannot be resolved the
        column is skipped with a warning, as are any other non-tensor columns.
        The per-frame instruction is taken from a string ``task`` column, or
        resolved from an integer ``task_index`` joined against the dataset's
        task table (the common LeRobot layout).
        """
        hf_dataset = dataset.hf_dataset.with_format("torch")
        columns: dict[str, Any] = {}
        # Video features are not part of the parquet files (the frames live in
        # the MP4s only), so they are resolved separately from the metadata.
        for key in dataset.video_keys:
            ref = LeRobotExperienceReplay._build_video_ref(dataset, key)
            if ref is not None:
                columns[key] = ref
        for name in hf_dataset.column_names:
            if name in columns:
                continue
            column = hf_dataset[name]
            if isinstance(column, torch.Tensor):
                columns[name] = column
                continue
            # recent `datasets` versions return a lazy Column: materialize it
            values = list(column)
            if values and isinstance(values[0], str):
                columns[name] = values
                continue
            try:
                if values and isinstance(values[0], torch.Tensor):
                    columns[name] = torch.stack(values)
                else:
                    columns[name] = torch.as_tensor(values)
            except (TypeError, ValueError, RuntimeError):
                torchrl_logger.warning(
                    f"LeRobotExperienceReplay: skipping column {name!r} which "
                    "could not be converted to a tensor or a VideoClipRef."
                )
        if "task" not in columns and "task_index" in columns:
            tasks = dataset.tasks
            if tasks:
                columns["task"] = [
                    str(tasks[int(i)]) for i in columns["task_index"].tolist()
                ]
        return columns

    @staticmethod
    def _build_video_ref(dataset: _LeRobotSnapshot, key: str) -> VideoClipRef | None:
        """Best-effort lazy :class:`~torchrl.data.VideoClipRef` for a video column.

        Spans the video files of ``key`` in episode order so that frame ``j`` of
        the reference lines up with row ``j`` of the (episode-major) dataset.
        Returns ``None`` on any failure so the caller falls back to skipping
        the column.
        """
        try:
            paths, counts = dataset.video_segments(key)
            return VideoClipRef.from_files(paths, num_frames_per_file=counts)
        except Exception as err:
            torchrl_logger.warning(
                f"LeRobotExperienceReplay: could not build a VideoClipRef for video "
                f"column {key!r} ({type(err).__name__}: {err}); skipping it."
            )
            return None
