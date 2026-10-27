# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import importlib.util
import shutil
from collections.abc import Mapping
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from tensordict import MetaData, TensorDict, TensorDictBase, UnbatchedTensor
from tensordict.utils import NestedKey, unravel_key
from torch.utils.data import DataLoader, Dataset
from torchrl._utils import logger as torchrl_logger
from torchrl.data.datasets._adapter_utils import _validate_export_dtypes
from torchrl.data.datasets.lerobot import _DEFAULT_KEY_MAP, _map_lerobot_key
from torchrl.data.replay_buffers.dataloader import tensordict_collate

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = Any

__all__ = ["LeRobotDatasetAdapter"]

_has_lerobot = importlib.util.find_spec("lerobot") is not None
_RESERVED = {"index", "episode_index", "frame_index", "timestamp", "task_index", "task"}


class _LeRobotDataset(Dataset[TensorDictBase]):
    def __init__(self, dataset: LeRobotDataset, key_map: dict[str, NestedKey]) -> None:
        self.dataset = dataset
        self.key_map = key_map

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TensorDictBase:
        values = self.dataset[index]
        result = TensorDict({}, [])
        for name, value in values.items():
            key = _map_lerobot_key(name, self.key_map)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(key, tuple)
                and key[0] == "next"
                and value.ndim == 0
            ):
                value = value.unsqueeze(-1)
            result.set(key, value)
        return result


class LeRobotDatasetAdapter:
    """Exchange trajectory data between TorchRL and LeRobot.

    Use this adapter to save demonstrations or replay data as LeRobot datasets,
    or to read existing LeRobot recordings as TensorDict frames.

    Export writes the fields selected by ``features``, including optional
    images or videos, to a new local dataset. The returned native
    ``LeRobotDataset`` can be uploaded with its ``push_to_hub`` method.

    Import returns a lazy dataset that reads frames through LeRobot, preserving
    its formatting and transforms. Use it with a PyTorch DataLoader and
    :func:`~torchrl.data.tensordict_collate` to read TensorDict batches.

    For export, provide ``repo_id``, ``fps`` and ``features``. Frames
    must be grouped by episode, with consecutive steps within each episode.
    :meth:`~torchrl.data.ReplayBuffer.export` can order them using
    ``trajectory_key`` and ``step_key``.

    Each export requires a new destination directory. Retained episode fragments
    are renumbered from frame zero, with timestamps generated from ``fps``.
    Next observations and terminal flags must be supplied explicitly.

    For import only, construct the adapter without export settings and call
    :meth:`from_external` with a finalized ``LeRobotDataset``. Temporal windows
    configured through ``delta_timestamps`` are not supported.

    Keyword Args:
        repo_id (str or None, optional): Hub repository identifier, e.g.
            ``"user/robot-demos"``. Defaults to ``None``; required for export.
            Used as metadata and does not trigger an upload.
        root (str, Path or None, optional): destination directory passed to
            LeRobot. Defaults to ``None``, using LeRobot's default location.
        fps (int or None, optional): positive recording frame rate. Defaults
            to ``None``; required for export.
        features (Mapping[str, dict] or None, optional): LeRobot schema mapping
            column names to their feature definitions. Only these columns are
            exported; LeRobot adds its generated metadata. Defaults to ``None``;
            required for export. An empty mapping exports only generated metadata.
            Camera tensors may use CHW or HWC layout. CHW tensors are permuted
            when the declared feature shape is HWC.
        key_map (Mapping[str, NestedKey] or None, optional): fixed overrides from
            LeRobot column names to source and imported TensorDict keys. Other
            columns use the canonical mapping of
            :func:`~torchrl.data.datasets.lerobot_columns_to_tensordict`.
            Defaults to ``None``. Duplicate or overlapping target keys are refused.
            The source episode ID and step number use the mappings for
            ``episode_index`` and ``frame_index``, respectively (by default
            ``"episode"`` and ``"frame"``). These are not exported as features.
        task (str or None, optional): fixed language instruction for every
            exported frame. Defaults to ``None``; export requires either
            ``task`` or ``task_key``.
        task_key (NestedKey or None, optional): source string instruction key,
            overriding ``task`` per frame when set. Defaults to ``None``.
        robot_type (str or None, optional): robot name stored in metadata.
            Defaults to ``None``.
        batch_size (int, optional): maximum frames fetched at once. Defaults
            to 256. Reduce for large cameras to limit host staging memory.
        video_codec (str, optional): native LeRobot video encoder name.
            Defaults to ``"h264"`` for browser playback.
            Choose another supported codec if your PyAV build lacks H.264.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from pathlib import Path
        >>> from tensordict import TensorDict
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torchrl.data.datasets import LeRobotDatasetAdapter
        >>> replay = ReplayBuffer(storage=LazyTensorStorage(4))
        >>> _ = replay.extend(TensorDict({
        ...     "action": torch.arange(4, dtype=torch.float32).unsqueeze(-1),
        ...     "episode": torch.tensor([1, 0, 1, 0]),
        ...     "frame": torch.tensor([1, 1, 0, 0]),
        ... }, [4]))
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     adapter = LeRobotDatasetAdapter(
        ...         repo_id="local/example", root=Path(directory) / "dataset", fps=30,
        ...         features={"action": {"dtype": "float32", "shape": (1,)}},
        ...         task="Move forward.",
        ...     )
        ...     native = replay.export(adapter, trajectory_key="episode", step_key="frame")
        ...     actions = [adapter.from_external(native)[i]["action"].item() for i in range(4)]
        >>> actions
        [3.0, 1.0, 2.0, 0.0]
        >>> native.push_to_hub(private=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        repo_id: str | None = None,
        root: str | Path | None = None,
        fps: int | None = None,
        features: Mapping[str, dict[str, Any]] | None = None,
        key_map: Mapping[str, NestedKey] | None = None,
        task: str | None = None,
        task_key: NestedKey | None = None,
        robot_type: str | None = None,
        batch_size: int = 256,
        video_codec: str = "h264",
    ) -> None:
        if not _has_lerobot:
            raise ImportError(
                "LeRobotDatasetAdapter requires LeRobot's dataset dependencies and "
                "Python >= 3.12. Install TorchRL with `pip install 'torchrl[lerobot]'`."
            )
        try:
            from lerobot.configs import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise ImportError(
                "LeRobotDatasetAdapter requires LeRobot >= 0.6.1 with its dataset "
                "dependencies. Install `pip install 'torchrl[lerobot]'`."
            ) from error

        self.features = copy.deepcopy(dict(features)) if features is not None else None
        self.key_map = {**_DEFAULT_KEY_MAP, **(key_map or {})}
        self._check_keys(tuple(dict.fromkeys((*(self.features or {}), *_RESERVED))))
        self.repo_id = repo_id
        self.root = root
        self.fps = fps
        self.episode_key = _map_lerobot_key("episode_index", self.key_map)
        self.step_key = _map_lerobot_key("frame_index", self.key_map)
        self.task = task
        self.task_key = task_key
        self._source_keys = tuple(
            dict.fromkeys(
                [
                    *(
                        _map_lerobot_key(name, self.key_map)
                        for name in (self.features or {})
                    ),
                    self.episode_key,
                    self.step_key,
                    *([task_key] if task_key is not None else []),
                ]
            )
        )
        self.robot_type = robot_type
        self.batch_size = batch_size
        self.video_codec = video_codec
        self._dataset_cls = LeRobotDataset
        self._encoder_cls = RGBEncoderConfig

    def _check_keys(self, names: tuple[str, ...]) -> None:
        targets = [unravel_key(_map_lerobot_key(name, self.key_map)) for name in names]
        paths = [(key,) if isinstance(key, str) else key for key in targets]
        if any(
            a == b[: len(a)] or b == a[: len(b)]
            for i, a in enumerate(paths)
            for b in paths[i + 1 :]
        ):
            raise ValueError(
                "key_map contains duplicate or overlapping TensorDict keys."
            )

    def _collate(self, batch: Any, *, dtypes: dict[NestedKey, torch.dtype]) -> Any:
        # Ignore unexported fields before collation can mix their keys or shapes.
        if isinstance(batch, list):
            batch = [
                item.select(*self._source_keys)
                if isinstance(item, TensorDictBase)
                else item
                for item in batch
            ]
        elif isinstance(batch, TensorDictBase):
            batch = batch.select(*self._source_keys)
        _validate_export_dtypes(batch, dtypes=dtypes)
        return tensordict_collate(batch)

    def _columns(self, batch: TensorDictBase) -> dict[str, Any]:
        if not isinstance(batch, TensorDict) or batch.ndim != 1:
            raise ValueError(
                "LeRobot export requires dense, unbatched TensorDict source items."
            )
        columns = {}
        for name, feature in self.features.items():
            value = batch.get(_map_lerobot_key(name, self.key_map))
            if isinstance(value, UnbatchedTensor):
                raise TypeError(
                    f"Feature {name!r} does not support UnbatchedTensor; convert it to a batched tensor."
                )
            if isinstance(value, MetaData):
                columns[name] = [value.data] * len(batch)
            elif isinstance(value, torch.Tensor):
                # LeRobot writes host arrays. Move user-provided tensors in batches.
                try:
                    array = value.detach().cpu().numpy()
                except (TypeError, RuntimeError) as error:
                    raise TypeError(
                        f"Cannot convert LeRobot feature {name!r} with dtype {value.dtype} to NumPy."
                    ) from error
                shape = tuple(feature["shape"])
                # TODO: Revisit layout handling using feature dimension names and an
                # explicit source layout for ambiguous shapes. Shape alone cannot always
                # distinguish CHW from HWC.
                if (
                    feature["dtype"] in ("image", "video")
                    and len(shape) == 3
                    and array.ndim == 4
                    and tuple(array.shape[1:]) != shape
                    and tuple(array.shape[1:]) == (shape[2], shape[0], shape[1])
                ):
                    array = array.transpose(0, 2, 3, 1)
                if shape == (1,) and array.ndim == 1:
                    array = array[:, None]
                if feature["dtype"] not in ("image", "video"):
                    # LeRobot retains numeric rows until save_episode().
                    array = array.copy()
                columns[name] = array
            else:
                columns[name] = value.tolist()
        return columns

    def to_external(self, dataset: Dataset[TensorDictBase], /) -> LeRobotDataset:
        """Write ordered TensorDict frames to a new finalized LeRobot dataset.

        Args:
            dataset (Dataset[TensorDictBase]): finite map-style source with
                unbatched frames, ordered by episode ID and consecutive step.

        Returns:
            lerobot.datasets.lerobot_dataset.LeRobotDataset: finalized native
            dataset backed by ``root``. No source tensors are retained.
        """
        if self.repo_id is None or self.fps is None or self.features is None:
            raise ValueError("Export requires repo_id, fps, and features.")
        if self.task is None and self.task_key is None:
            raise ValueError("Export requires either task or task_key.")
        if not len(dataset):
            raise ValueError("Cannot export an empty LeRobot dataset.")
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=partial(self._collate, dtypes={}),
            num_workers=0,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        native = None
        previous_episode = previous_step = None
        try:
            for batch in loader:
                columns = self._columns(batch)
                metadata = []
                for key in (self.episode_key, self.step_key):
                    value = batch.get(key)
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.dtype
                        not in (
                            torch.int8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                            torch.uint8,
                        )
                        or value.shape not in (batch.shape, (*batch.shape, 1))
                    ):
                        raise ValueError(f"{key!r} must contain one integer per frame.")
                    metadata.append(value.detach().cpu().numpy().reshape(-1))
                episodes, steps = metadata
                if (steps < 0).any():
                    raise ValueError("Frame steps must be non-negative.")
                if self.task_key is None:
                    tasks = [self.task] * len(batch)
                else:
                    value = batch.get(self.task_key)
                    tasks = (
                        [value.data] * len(batch)
                        if isinstance(value, MetaData)
                        else value.tolist()
                    )
                if native is None:
                    native = self._dataset_cls.create(
                        repo_id=self.repo_id,
                        root=self.root,
                        fps=self.fps,
                        features=self.features,
                        robot_type=self.robot_type,
                        video_backend="pyav",
                        # The native streaming path drops frames on a full queue.
                        streaming_encoding=False,
                        image_writer_processes=0,
                        image_writer_threads=0,
                        rgb_encoder=(
                            self._encoder_cls(vcodec=self.video_codec)
                            if any(
                                feature["dtype"] == "video"
                                for feature in self.features.values()
                            )
                            else None
                        ),
                    )
                for i, (episode, step) in enumerate(
                    zip(episodes.tolist(), steps.tolist())
                ):
                    if previous_episode is not None:
                        if episode != previous_episode:
                            if episode <= previous_episode:
                                raise ValueError(
                                    "Episodes must be contiguous and ordered by increasing ID."
                                )
                            native.save_episode(parallel_encoding=False)
                        elif step != previous_step + 1:
                            raise ValueError(
                                "Steps within each episode must be consecutive."
                            )
                    native.add_frame(
                        {
                            **{name: array[i] for name, array in columns.items()},
                            "task": tasks[i],
                        }
                    )
                    previous_episode, previous_step = episode, step
            native.save_episode(parallel_encoding=False)
            native.finalize()
        except BaseException:
            if native is not None:
                # Close writers before removing the incomplete dataset.
                with suppress(Exception):
                    native.clear_episode_buffer()
                with suppress(Exception):
                    native.finalize()
                try:
                    shutil.rmtree(native.root)
                except OSError as cleanup_error:
                    torchrl_logger.warning(
                        "Could not remove incomplete LeRobot dataset %s: %s",
                        native.root,
                        cleanup_error,
                    )
            raise
        return native

    def from_external(self, dataset: LeRobotDataset, /) -> Dataset[TensorDictBase]:
        """Expose native LeRobot frames through a lazy TensorDict dataset.

        Args:
            dataset (lerobot.datasets.lerobot_dataset.LeRobotDataset): finalized
                native dataset with no ``delta_timestamps`` windows. Its numeric
                formatting, camera decoding options and image transforms apply
                on access. Keep the dataset and its backing files fixed.

        Returns:
            Dataset[TensorDictBase]: one unbatched TensorDict per native frame,
            with canonical or explicitly mapped keys and decoded camera tensors.
        """
        if not isinstance(dataset, self._dataset_cls):
            raise TypeError("dataset must be a native LeRobotDataset.")
        if dataset.delta_timestamps:
            raise ValueError(
                "Import requires individual frames; disable delta_timestamps on the native dataset."
            )
        self._check_keys(tuple(dict.fromkeys((*dataset.features, "task"))))
        return _LeRobotDataset(dataset, dict(self.key_map))
