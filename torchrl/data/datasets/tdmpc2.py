# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os.path
import re
import shutil
import tempfile
import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.data.datasets.common import BaseDatasetExperienceReplay
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer

_has_tqdm = importlib.util.find_spec("tqdm", None) is not None
_has_hf_hub = importlib.util.find_spec("huggingface_hub", None) is not None


class TDMPC2ExperienceReplay(BaseDatasetExperienceReplay):
    """TD-MPC2 multi-task offline dataset.

    This class downloads the TD-MPC2 ``mt30`` and ``mt80`` datasets and
    converts them to TorchRL TED format.

    Source repository: https://github.com/nicklashansen/tdmpc2
    Dataset: https://huggingface.co/datasets/nicklashansen/tdmpc2

    Args:
        dataset_id (str): Dataset identifier. Must be one of ``"mt30"`` or
            ``"mt80"``.
        batch_size (int): Batch-size used during sampling.

    Keyword Args:
        root (Path or str, optional): The dataset root directory. The processed
            memory-mapped data is saved under ``<root>/<dataset_id>``. If none
            is provided, it defaults to ``~/.cache/torchrl/tdmpc2``.
        download (bool or str, optional): Whether the dataset should be
            downloaded if not found. Defaults to ``True``. Download can also be
            passed as ``"force"``, in which case downloaded data is overwritten.
        sampler (Sampler, optional): Sampler used during sampling.
        writer (Writer, optional): Writer used during sampling. If none is
            provided, :class:`~torchrl.data.replay_buffers.writers.ImmutableDatasetWriter`
            is used.
        collate_fn (callable, optional): Collate function used for batching.
        pin_memory (bool): Whether pin_memory() should be called on samples.
        prefetch (int, optional): Number of prefetched batches.
        transform (Transform, optional): Transform executed when sample() is called.
        split_trajs (bool, optional): If ``True``, the dataset is split into
            trajectories and saved under ``<root>/<dataset_id>_split``.

    """

    available_datasets = ["mt30", "mt80"]
    _HF_REPO_ID = "nicklashansen/tdmpc2"
    _EXPECTED_NUM_CHUNKS = {"mt30": 4, "mt80": 20}

    def __init__(
        self,
        dataset_id,
        batch_size: int,
        *,
        root: str | Path | None = None,
        download: bool = True,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: torchrl.envs.Transform | None = None,  # noqa-F821
        split_trajs: bool = False,
    ):
        if dataset_id not in self.available_datasets:
            raise ValueError(
                f"The dataset_id {dataset_id} isn't part of the accepted datasets. "
                f"Use one of {self.available_datasets}."
            )
        self.dataset_id = dataset_id
        self.split_trajs = split_trajs
        self.download = download

        if root is None:
            root = _get_root_dir("tdmpc2")
            os.makedirs(root, exist_ok=True)
        self.root = root

        if self.download == "force" or (self.download and not self._is_downloaded()):
            if self.download == "force":
                try:
                    if os.path.exists(self.data_path_root):
                        shutil.rmtree(self.data_path_root)
                    if self.data_path != self.data_path_root:
                        shutil.rmtree(self.data_path)
                except FileNotFoundError:
                    pass
            storage = self._download_and_preproc()
        elif self.split_trajs and not os.path.exists(self.data_path):
            storage = self._make_split()
        else:
            storage = self._load()
        storage = TensorStorage(storage)

        if writer is None:
            writer = ImmutableDatasetWriter()

        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
        )

    @staticmethod
    def _chunk_idx(path: str | Path) -> int:
        match = re.search(r"chunk_(\d+)\.pt$", Path(path).as_posix())
        if match is None:
            raise ValueError(f"Could not parse TD-MPC2 chunk index from path {path}.")
        return int(match.group(1))

    @classmethod
    def _sort_chunk_paths(cls, chunk_paths: Sequence[str | Path]) -> list[Path]:
        return sorted((Path(path) for path in chunk_paths), key=cls._chunk_idx)

    @staticmethod
    def _get_optional_key(td: TensorDictBase, *keys: str):
        for key in keys:
            if key in td.keys():
                return td.get(key)
        return None

    @classmethod
    def _get_required_key(cls, td: TensorDictBase, *keys: str):
        out = cls._get_optional_key(td, *keys)
        if out is None:
            raise KeyError(f"Could not find any of {keys} in chunk keys {td.keys()}.")
        return out

    @classmethod
    def _load_chunk(cls, path: str | Path) -> TensorDictBase:
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(chunk, TensorDictBase):
            return chunk
        if isinstance(chunk, dict):
            return TensorDict.from_dict(chunk, auto_batch_size=True)
        raise TypeError(
            f"Expected chunk {path} to contain a TensorDict or dict, got {type(chunk)}."
        )

    @classmethod
    def _count_chunk_steps(
        cls,
        chunk: TensorDictBase,
    ) -> tuple[int, int]:
        obs = cls._get_required_key(chunk, "obs", "observation")
        if not isinstance(obs, torch.Tensor):
            raise TypeError(
                f"Expected obs to be a tensor, but got type {type(obs)} instead."
            )
        if obs.ndim < 2:
            raise RuntimeError(
                f"Expected obs with shape [E, T, ...], but got shape {obs.shape}."
            )
        num_episodes, horizon = obs.shape[:2]
        if horizon < 2:
            raise RuntimeError(
                f"Expected horizon >= 2 for conversion, but got shape {obs.shape}."
            )
        return num_episodes * (horizon - 1), num_episodes

    @classmethod
    def _convert_chunk(
        cls, chunk: TensorDictBase, episode_offset: int
    ) -> tuple[TensorDict, int]:
        obs = cls._get_required_key(chunk, "obs", "observation")
        action = cls._get_required_key(chunk, "action", "actions")
        reward = cls._get_required_key(chunk, "reward", "rewards")
        terminated = cls._get_optional_key(chunk, "terminated", "terminals")

        if not isinstance(obs, torch.Tensor):
            raise TypeError(
                f"Expected obs to be a tensor, but got type {type(obs)} instead."
            )
        if not isinstance(action, torch.Tensor):
            raise TypeError(
                f"Expected action to be a tensor, but got type {type(action)} instead."
            )
        if not isinstance(reward, torch.Tensor):
            raise TypeError(
                f"Expected reward to be a tensor, but got type {type(reward)} instead."
            )
        if terminated is not None and not isinstance(terminated, torch.Tensor):
            raise TypeError(
                "Expected terminated to be a tensor if present, "
                f"but got type {type(terminated)} instead."
            )

        num_episodes, horizon = obs.shape[:2]
        if action.shape[:2] != (num_episodes, horizon):
            raise RuntimeError(
                f"Mismatching action shape {action.shape} for obs shape {obs.shape}."
            )
        if reward.shape[:2] != (num_episodes, horizon):
            raise RuntimeError(
                f"Mismatching reward shape {reward.shape} for obs shape {obs.shape}."
            )
        if terminated is not None and terminated.shape[:2] != (num_episodes, horizon):
            raise RuntimeError(
                "Mismatching terminated shape "
                f"{terminated.shape} for obs shape {obs.shape}."
            )

        transition_length = horizon - 1
        observation = obs[:, :-1]
        next_observation = obs[:, 1:]
        action = action[:, 1:]

        reward = reward[:, 1:]
        if reward.ndim == 2:
            reward = reward.unsqueeze(-1)

        if terminated is None:
            terminated = torch.zeros(
                num_episodes, transition_length, 1, dtype=torch.bool
            )
        else:
            terminated = terminated[:, 1:]
            if terminated.ndim == 2:
                terminated = terminated.unsqueeze(-1)
            terminated = terminated.bool()

        truncated = torch.zeros_like(terminated)
        truncated[:, -1] = ~terminated[:, -1]
        done = terminated | truncated

        episode = torch.arange(
            episode_offset,
            episode_offset + num_episodes,
            dtype=torch.int64,
        ).unsqueeze(-1)
        episode = episode.expand(-1, transition_length)

        td = TensorDict(
            {
                "observation": observation,
                "action": action,
                "episode": episode,
                "done": torch.zeros_like(done),
                "terminated": torch.zeros_like(terminated),
                "truncated": torch.zeros_like(truncated),
                "next": TensorDict(
                    {
                        "observation": next_observation,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                    batch_size=(num_episodes, transition_length),
                ),
            },
            batch_size=(num_episodes, transition_length),
        )

        task = cls._get_optional_key(chunk, "task")
        if isinstance(task, torch.Tensor):
            if task.shape[:2] == (num_episodes, horizon):
                td.set("task", task[:, 1:])
            elif task.shape[0] == num_episodes:
                td.set(
                    "task",
                    task.unsqueeze(1).expand(num_episodes, transition_length, *task.shape[1:]),
                )
            else:
                warnings.warn(
                    f"Skipping task key with shape {task.shape}: expected shape "
                    f"[E, T, ...] or [E, ...] with E={num_episodes} and T={horizon}."
                )

        return td.reshape(-1), num_episodes

    def _download_from_huggingface(self, tempdir: str | Path) -> list[Path]:
        if not _has_hf_hub:
            raise ImportError(
                "huggingface_hub is required for TDMPC2 dataset download."
            )
        from huggingface_hub import hf_hub_download, HfApi

        dataset = HfApi().dataset_info(self._HF_REPO_ID)

        chunk_files = []
        for sibling in dataset.siblings:
            rfilename = sibling.rfilename
            if rfilename.startswith(f"{self.dataset_id}/chunk_") and rfilename.endswith(
                ".pt"
            ):
                chunk_files.append(rfilename)
        if not chunk_files:
            raise RuntimeError(
                f"Could not find TD-MPC2 chunk files for dataset {self.dataset_id}."
            )

        chunk_files = self._sort_chunk_paths(chunk_files)
        expected = self._EXPECTED_NUM_CHUNKS[self.dataset_id]
        if len(chunk_files) != expected:
            warnings.warn(
                f"Expected {expected} chunk files for {self.dataset_id} but found "
                f"{len(chunk_files)} files."
            )

        data_cache_dir = Path(tempdir) / "data"
        out = []
        for chunk_path in chunk_files:
            local_path = hf_hub_download(
                self._HF_REPO_ID,
                subfolder=str(chunk_path.parent),
                filename=chunk_path.name,
                repo_type="dataset",
                cache_dir=str(data_cache_dir),
            )
            out.append(Path(local_path))
        return out

    @classmethod
    def _preproc_chunks(
        cls, chunk_paths: Sequence[str | Path], data_path: str | Path
    ) -> TensorDict:
        chunk_paths = cls._sort_chunk_paths(chunk_paths)
        if not chunk_paths:
            raise RuntimeError("No TD-MPC2 chunk files found.")

        total_steps = 0
        total_episodes = 0
        for chunk_path in chunk_paths:
            chunk = cls._load_chunk(chunk_path)
            steps, episodes = cls._count_chunk_steps(chunk)
            total_steps += steps
            total_episodes += episodes

        torchrl_logger.info(
            f"Found {len(chunk_paths)} chunks with {total_episodes} episodes and "
            f"{total_steps} transitions."
        )

        first_chunk = cls._load_chunk(chunk_paths[0])
        first_td, first_num_episodes = cls._convert_chunk(first_chunk, episode_offset=0)

        data_path = Path(data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        td_data = first_td[0].expand(total_steps).memmap_like(data_path, num_threads=32)

        if _has_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=total_steps)
        else:
            pbar = None

        idx = 0
        episode_offset = 0
        with pbar if pbar is not None else nullcontext():
            for i, chunk_path in enumerate(chunk_paths):
                if i == 0:
                    td_chunk = first_td
                    num_episodes = first_num_episodes
                else:
                    chunk = cls._load_chunk(chunk_path)
                    td_chunk, num_episodes = cls._convert_chunk(
                        chunk, episode_offset=episode_offset
                    )

                next_idx = idx + td_chunk.shape[0]
                td_data[idx:next_idx] = td_chunk
                idx = next_idx
                episode_offset += num_episodes

                if pbar is not None:
                    pbar.update(td_chunk.shape[0])
                    pbar.set_description(f"chunk={Path(chunk_path).name}")

        if idx != total_steps:
            raise RuntimeError(
                f"Dataset writing failed: expected {total_steps} transitions but "
                f"wrote {idx} transitions."
            )
        return td_data

    def _download_and_preproc(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chunk_paths = self._download_from_huggingface(tempdir)
            return self._preproc_chunks(chunk_paths, self.data_path_root)

    def _make_split(self):
        from torchrl.collectors.utils import split_trajectories

        td_data = TensorDict.load_memmap(self.data_path_root)
        td_data = split_trajectories(td_data).memmap_(self.data_path)
        return td_data

    def _load(self):
        return TensorDict.load_memmap(self.data_path)

    @property
    def data_path(self):
        if self.split_trajs:
            return Path(self.root) / (self.dataset_id + "_split")
        return self.data_path_root

    @property
    def data_path_root(self):
        return Path(self.root) / self.dataset_id

    def _is_downloaded(self):
        return os.path.exists(self.data_path_root)
