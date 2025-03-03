# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os.path
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
from tensordict import PersistentTensorDict, TensorDict
from torchrl._utils import (
    KeyDependentDefaultDict,
    logger as torchrl_logger,
    print_directory_tree,
)
from torchrl.data.datasets.common import BaseDatasetExperienceReplay
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer

_has_tqdm = importlib.util.find_spec("tqdm", None) is not None
_has_h5py = importlib.util.find_spec("h5py", None) is not None
_has_hf_hub = importlib.util.find_spec("huggingface_hub", None) is not None

_NAME_MATCH = KeyDependentDefaultDict(lambda key: key)
_NAME_MATCH["observations"] = "observation"
_NAME_MATCH["rewards"] = "reward"
_NAME_MATCH["actions"] = "action"
_NAME_MATCH["env_infos"] = "info"


class RobosetExperienceReplay(BaseDatasetExperienceReplay):
    """Roboset experience replay dataset.

    This class downloads the H5 data from roboset and processes it in a mmap
    format, which makes indexing (and therefore sampling) faster.

    Learn more about roboset here: https://sites.google.com/view/robohive/roboset

    The data format follows the :ref:`TED convention <TED-format>`.

    Args:
        dataset_id (str): the dataset to be downloaded. Must be part of RobosetExperienceReplay.available_datasets.
        batch_size (int): Batch-size used during sampling. Can be overridden by `data.sample(batch_size)` if
            necessary.

    Keyword Args:
        root (Path or str, optional): The Roboset dataset root directory.
            The actual dataset memory-mapped files will be saved under
            `<root>/<dataset_id>`. If none is provided, it defaults to
            `~/.cache/torchrl/atari`.roboset`.
        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as ``"force"``,
            in which case the downloaded data will be overwritten.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.writers.ImmutableDatasetWriter` will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :class:`~torchrl.envs.transforms.transforms.Compose` class.
        split_trajs (bool, optional): if ``True``, the trajectories will be split
            along the first dimension and padded to have a matching shape.
            To split the trajectories, the ``"done"`` signal will be used, which
            is recovered via ``done = truncated | terminated``. In other words,
            it is assumed that any ``truncated`` or ``terminated`` signal is
            equivalent to the end of a trajectory.
            Defaults to ``False``.

    Attributes:
        available_datasets: a list of accepted entries to be downloaded.

    Examples:
        >>> import torch
        >>> torch.manual_seed(0)
        >>> from torchrl.envs.transforms import ExcludeTransform
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> d = RobosetExperienceReplay("FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4", batch_size=32,
        ...     transform=ExcludeTransform("info", ("next", "info")))  # excluding info dict for conciseness
        >>> for batch in d:
        ...     break
        >>> # data is organised by seed and episode, but stored contiguously
        >>> print(f"{batch['seed']}, {batch['episode']}")
        tensor([2, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 0, 2, 0, 2, 2, 1,
                0, 2, 0, 0, 1, 1, 2, 1]) tensor([17, 20, 18,  9,  6,  1, 12,  6,  2,  6,  8, 15,  8, 21, 17,  3,  9, 20,
                23, 12,  3, 16, 19, 16, 16,  4,  4, 12,  1,  2, 15, 24])
        >>> print(batch)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([32, 9]), device=cpu, dtype=torch.float64, is_shared=False),
                done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                episode: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False),
                index: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([32, 75]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([32, 75]), device=cpu, dtype=torch.float64, is_shared=False),
                seed: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                time: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float64, is_shared=False)},
                truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([32]),
            device=cpu,
            is_shared=False)

    """

    available_datasets = [
        "DAPG(expert)/door_v2d-v1",
        "DAPG(expert)/relocate_v2d-v1",
        "DAPG(expert)/hammer_v2d-v1",
        "DAPG(expert)/pen_v2d-v1",
        "DAPG(human)/door_v2d-v1",
        "DAPG(human)/relocate_v2d-v1",
        "DAPG(human)/hammer_v2d-v1",
        "DAPG(human)/pen_v2d-v1",
        "FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4",
        "FK1-v4(expert)/FK1_Knob2OffRandom_v2d-v4",
        "FK1-v4(expert)/FK1_LdoorOpenRandom_v2d-v4",
        "FK1-v4(expert)/FK1_SdoorOpenRandom_v2d-v4",
        "FK1-v4(expert)/FK1_Knob1OnRandom_v2d-v4",
        "FK1-v4(human)/human_demos_by_playdata",
        "FK1-v4(human)/human_demos_by_task/human_demo_singleTask_Fixed-v4",
        "FK1-v4(human)/human_demos_by_task/FK1_SdoorOpenRandom_v2d-v4",
        "FK1-v4(human)/human_demos_by_task/FK1_LdoorOpenRandom_v2d-v4",
        "FK1-v4(human)/human_demos_by_task/FK1_Knob2OffRandom_v2d-v4",
        "FK1-v4(human)/human_demos_by_task/FK1_Knob1OnRandom_v2d-v4",
        "FK1-v4(human)/human_demos_by_task/FK1_MicroOpenRandom_v2d-v4",
    ]

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
        **env_kwargs,
    ):
        if not _has_h5py or not _has_hf_hub:
            raise ImportError(
                "h5py and huggingface_hub are required for Roboset datasets."
            )
        if dataset_id not in self.available_datasets:
            raise ValueError(
                f"The dataset_id {dataset_id} isn't part of the accepted datasets. "
                f"To check which dataset can be downloaded, call `{type(self)}.available_datasets`."
            )
        self.dataset_id = dataset_id
        if root is None:
            root = _get_root_dir("roboset")
            os.makedirs(root, exist_ok=True)
        self.root = root
        self.split_trajs = split_trajs
        self.download = download
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

    def _download_from_huggingface(self, tempdir):
        try:
            from huggingface_hub import hf_hub_download, HfApi
        except ImportError:
            raise ImportError(
                f"huggingface_hub is required for downloading {type(self)}'s datasets."
            )
        dataset = HfApi().dataset_info("jdvakil/RoboSet_Sim")
        h5_files = []
        datapath = Path(tempdir) / "data"
        for sibling in dataset.siblings:
            if sibling.rfilename.startswith(
                self.dataset_id
            ) and sibling.rfilename.endswith(".h5"):
                path = Path(sibling.rfilename)
                local_path = hf_hub_download(
                    "jdvakil/RoboSet_Sim",
                    subfolder=str(path.parent),
                    filename=str(path.parts[-1]),
                    repo_type="dataset",
                    cache_dir=str(datapath),
                )
                h5_files.append(local_path)

        return sorted(h5_files)

    def _download_and_preproc(self):

        with tempfile.TemporaryDirectory() as tempdir:
            h5_data_files = self._download_from_huggingface(tempdir)
            return self._preproc_h5(h5_data_files)

    def _preproc_h5(self, h5_data_files):
        td_data = TensorDict()
        total_steps = 0
        torchrl_logger.info(
            f"first read through data files {h5_data_files} to create data structure..."
        )
        episode_dict = {}
        h5_datas = []
        for seed, h5_data_name in enumerate(h5_data_files):
            torchrl_logger.info(f"\nReading {h5_data_name}")
            h5_data = PersistentTensorDict.from_h5(h5_data_name)
            h5_datas.append(h5_data)
            for i, (episode_key, episode) in enumerate(h5_data.items()):
                episode_num = int(episode_key[len("Trial") :])
                episode_len = episode["actions"].shape[0]
                episode_dict[(seed, episode_num)] = (episode_key, episode_len)
                # Get the total number of steps for the dataset
                total_steps += episode_len
                torchrl_logger.info(f"total_steps {total_steps}")
                if i == 0 and seed == 0:
                    td_data.set("episode", 0)
                    td_data.set("seed", 0)
                    for key, val in episode.items():
                        match = _NAME_MATCH[key]
                        if key in ("observations", "env_infos", "done"):
                            td_data.set(("next", match), torch.zeros_like(val[0]))
                            td_data.set(match, torch.zeros_like(val[0]))
                        elif key not in ("rewards",):
                            td_data.set(match, torch.zeros_like(val[0]))
                        else:
                            td_data.set(
                                ("next", match),
                                torch.zeros_like(val[0].unsqueeze(-1)),
                            )

        # give it the proper size
        td_data["next", "done"] = td_data["next", "done"].unsqueeze(-1)
        td_data["done"] = td_data["done"].unsqueeze(-1)
        td_data["next", "terminated"] = td_data["next", "done"]
        td_data["next", "truncated"] = td_data["next", "done"]
        td_data["terminated"] = td_data["done"]
        td_data["truncated"] = td_data["done"]

        td_data = td_data.expand(total_steps)
        # save to designated location
        torchrl_logger.info(f"creating tensordict data in {self.data_path_root}: ")
        td_data = td_data.memmap_like(self.data_path_root)
        # torchrl_logger.info(f"tensordict structure: {td_data}")
        torchrl_logger.info(
            f"Local dataset structure: {print_directory_tree(self.data_path_root)}"
        )

        torchrl_logger.info(f"Reading data from {len(episode_dict)} episodes")
        index = 0
        if _has_tqdm:
            from tqdm import tqdm
        else:
            tqdm = None
        with tqdm(total=total_steps) if _has_tqdm else nullcontext() as pbar:
            # iterate over episodes and populate the tensordict
            for seed, episode_num in sorted(episode_dict, key=lambda key: key[1]):
                h5_data = h5_datas[seed]
                episode_key, steps = episode_dict[(seed, episode_num)]
                episode = h5_data.get(episode_key)
                idx = slice(index, (index + steps))
                data_view = td_data[idx]
                data_view.fill_("episode", episode_num)
                data_view.fill_("seed", seed)
                for key, val in episode.items():
                    match = _NAME_MATCH[key]
                    if steps != val.shape[0]:
                        raise RuntimeError(
                            f"Mismatching number of steps for key {key}: was {steps} but got {val.shape[0]}."
                        )
                    if key in (
                        "observations",
                        "env_infos",
                    ):
                        data_view["next", match][:-1].copy_(val[1:])
                        data_view[match].copy_(val)
                    elif key not in ("rewards", "done", "terminated", "truncated"):
                        data_view[match].copy_(val)
                    elif key in ("done", "terminated", "truncated"):
                        data_view[match].copy_(val.unsqueeze(-1))
                        data_view[("next", match)].copy_(val.unsqueeze(-1))
                    else:
                        data_view[("next", match)].copy_(val.unsqueeze(-1))
                data_view["next", "terminated"].copy_(data_view["next", "done"])
                if pbar is not None:
                    pbar.update(steps)
                    pbar.set_description(
                        f"index={index} - episode num {episode_num} - seed {seed}"
                    )
                index += steps
        return td_data

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
