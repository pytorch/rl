# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os.path
import tempfile
from pathlib import Path
from typing import Callable

import torch
from tensordict import MemoryMappedTensor, PersistentTensorDict, TensorDict
from torchrl._utils import KeyDependentDefaultDict
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import Writer

_NAME_MATCH = KeyDependentDefaultDict(lambda key: key)
_NAME_MATCH["observations"] = "observation"
_NAME_MATCH["rewards"] = "reward"
_NAME_MATCH["truncations"] = "truncated"
_NAME_MATCH["terminations"] = "terminated"
_NAME_MATCH["actions"] = "action"
_NAME_MATCH["infos"] = "info"


class MinariExperienceReplay(TensorDictReplayBuffer):
    """Minari Experience replay dataset.

    Args:
        dataset_id (str):
        batch_size (int):

    Keyword Args:
        root (Path or str, optional):
        download (bool, optional):
    """
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
        transform: "torchrl.envs.Transform" | None = None,  # noqa-F821
        split_trajs: bool = False,
        **env_kwargs,
    ):
        self.dataset_id = dataset_id
        if root is None:
            root = _get_root_dir("minari")
            os.makedirs(root, exist_ok=True)
        self.root = root
        self.split_trajs = split_trajs
        self.download = download
        if self.download and not self._is_downloaded():
            storage = self._download_and_preproc()
        elif self.split_trajs and not os.path.exists(self.data_path):
            storage = self._make_split()
        else:
            storage = self._load()
        storage = TensorStorage(storage)
        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
        )

    def _is_downloaded(self):
        return os.path.exists(self.data_path)

    @property
    def data_path(self):
        if self.split_trajs:
            return Path(self.root) / (self.dataset_id + "_split")
        return self.data_path_root

    @property
    def data_path_root(self):
        return Path(self.root) / self.dataset_id

    def _download_and_preproc(self):
        import minari

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MINARI_DATASETS_PATH"] = tmpdir
            minari.download_dataset(dataset_id=self.dataset_id)
            minari.load_dataset(self.dataset_id)
            h5_data = PersistentTensorDict.from_h5(
                Path(tmpdir) / self.dataset_id / "data/main_data.hdf5"
            )

            # Get the total number of steps for the dataset
            total_steps = sum(
                h5_data[episode, "actions"].shape[0] for episode in h5_data.keys()
            )
            # populate the tensordict
            td_data = TensorDict({}, [])
            for key, episode in h5_data.items():
                for key, val in episode.items():
                    match = _NAME_MATCH[key]
                    if key in ("observations", "state"):
                        td_data.set(("next", match), torch.zeros_like(val)[0])
                        td_data.set(match, torch.zeros_like(val)[0])
                    elif key not in ("terminations", "truncations", "rewards"):
                        td_data.set(match, torch.zeros_like(val)[0])
                    else:
                        td_data.set(
                            ("next", match), torch.zeros_like(val)[0].unsqueeze(-1)
                        )
                break
            # give it the proper size
            td_data = td_data.expand(total_steps)
            # save to designated location
            td_data.memmap_(self.data_path_root)
            # iterate over episodes and populate the tensordict
            index = 0
            for key, episode in h5_data.items():
                for key, val in episode.items():
                    match = _NAME_MATCH[key]
                    if key in (
                        "observations",
                        "state",
                    ):
                        steps = val.shape[0] - 1
                        td_data["next", match][index : (index + steps)] = val[1:]
                        td_data[match][index : (index + steps)] = val[:-1]
                    elif key not in ("terminations", "truncations", "rewards"):
                        steps = val.shape[0]
                        td_data[match][index : (index + val.shape[0])] = val
                    else:
                        steps = val.shape[0]
                        td_data[("next", match)][
                            index : (index + val.shape[0])
                        ] = val.unsqueeze(-1)
                index += steps
            # Add a "done" entry
            with td_data.unlock_():
                td_data["next", "done"] = MemoryMappedTensor.from_tensor(
                    (td_data["next", "terminated"] | td_data["next", "truncated"])
                )
                if self.split_trajs:
                    from torchrl.objectives.utils import split_trajectories

                    td_data = split_trajectories(td_data).memmap_(self.data_path)
            return td_data

    def _make_split(self):
        from torchrl.objectives.utils import split_trajectories

        td_data = TensorDict.load_memmap(self.data_path_root)
        td_data = split_trajectories(td_data).memmap_(self.data_path)
        return td_data

    def _load(self):
        return TensorDict.load_memmap(self.data_path)
