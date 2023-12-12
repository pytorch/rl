# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import glob
import importlib.util
import json
import os.path
import shutil
import tempfile

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from tensordict import PersistentTensorDict, TensorDict
from torchrl._utils import KeyDependentDefaultDict, print_directory_tree
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import Writer
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

_has_tqdm = importlib.util.find_spec("tqdm", None) is not None

_NAME_MATCH = KeyDependentDefaultDict(lambda key: key)
_NAME_MATCH["observations"] = "observation"
_NAME_MATCH["rewards"] = "reward"
_NAME_MATCH["truncations"] = "truncated"
_NAME_MATCH["terminations"] = "terminated"
_NAME_MATCH["actions"] = "action"
_NAME_MATCH["infos"] = "info"


class RobosetExperienceReplay(TensorDictReplayBuffer):

    available_datasets = {
        "DAPG(expert)/door_v2d-v1": "1_gRk-k3S8aZortmaeoskeDH5oTo_zKYj",
        "DAPG(expert)/relocate_v2d-v1": "1QyPss3BUAdDfq5OI6v7HAwcszDteZN10",
        "DAPG(expert)/hammer_v2d-v1": "1NaipPfSsyCbxlg8Lw4EeJ7ZoWbmB8tX1",
        "DAPG(expert)/pen_v2d-v1": "11UyaAlbYJcMwR9DBNv9igx_icTkT0roB",
        "FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4": "1Y3q6BDR0cAVMOnPyPt0yQJXeIpcLiXBY",
        "FK1-v4(expert)/FK1_Knob2OffRandom_v2d-v4": "1qs3pI_PrFDA_KTCbsJYZ2SVQvO5rvjYf",
        "FK1-v4(expert)/FK1_LdoorOpenRandom_v2d-v4": "12jQNpZv7Zxb6q4Z38xPpJMPDj19zLuYI",
        "FK1-v4(expert)/FK1_SdoorOpenRandom_v2d-v4": "1zpUPeXMXdWbSJAS87k_6BgCEgvcyx228",
        "FK1-v4(expert)/FK1_Knob1OnRandom_v2d-v4": "1dWe_aAB-jPj-kgN5NYJX9-amHbM6TpXC",
    }

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
            root = _get_root_dir("roboset")
            os.makedirs(root, exist_ok=True)
        self.root = root
        self.split_trajs = split_trajs
        self.download = download
        if self.download == "force" or (self.download and not self._is_downloaded()):
            if self.download == "force":
                try:
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
        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
        )

    def _download_from_google_drive(self, tempdir):
        tempdir = Path(tempdir)
        datapath = tempdir / "data"
        os.makedirs(datapath, exist_ok=True)

        try:
            import gdown
        except ImportError:
            raise ImportError(
                f"gdown is required for downloading {type(self)}'s datasets."
            )

        url = f"https://drive.google.com/drive/folders/{self.available_datasets[self.dataset_id]}"
        gdown.download_folder(url, output=str(datapath), quiet=False)
        print_directory_tree(datapath)
        # find paths to all h5 files
        h5_files = []

        # Recursively search for .h5 files in the specified path
        for root, dirs, files in os.walk(datapath):
            h5_files.extend(glob.glob(os.path.join(root, "*.h5")))

        return sorted(h5_files)

    def _download_and_preproc(self):

        with tempfile.TemporaryDirectory() as tempdir:
            h5_data_files = self._download_from_google_drive(tempdir)
            return self._preproc_h5(h5_data_files)

    def _preproc_h5(self, h5_data_files):
        td_data = TensorDict({}, [])
        total_steps = 0
        print(
            f"first read through data files {h5_data_files} to create data structure..."
        )
        episode_dict = {}
        h5_datas = []
        for seed, h5_data_name in enumerate(h5_data_files):
            print("\nReading", h5_data_name)
            h5_data = PersistentTensorDict.from_h5(h5_data_name)
            h5_datas.append(h5_data)
            for i, (episode_key, episode) in enumerate(h5_data.items()):
                episode_num = int(episode_key[len("Trial") :])
                episode_len = episode["actions"].shape[0]
                episode_dict[(seed, episode_num)] = (episode_key, episode_len)
                # Get the total number of steps for the dataset
                total_steps += episode_len
                print("total_steps", total_steps, end="\t")
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
        td_data["next", "terminated"] = td_data["next", "done"]
        td_data["next", "truncated"] = td_data["next", "done"]

        td_data = td_data.expand(total_steps)
        # save to designated location
        print(f"creating tensordict data in {self.data_path_root}: ", end="\t")
        td_data = td_data.memmap_like(self.data_path_root)
        # print("tensordict structure:", td_data)
        print("Local dataset structure:", print_directory_tree(self.data_path_root))

        print(f"Reading data from {len(episode_dict)} episodes")
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
                    elif key not in ("rewards",):
                        data_view[match].copy_(val)
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
