# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gzip
import io
import pathlib
import shutil

import mmap
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import tqdm
from tensordict import NonTensorData, TensorDict, MemoryMappedTensor

from torchrl.data import LazyMemmapStorage, Storage, TensorDictReplayBuffer
from torchrl.envs.utils import _classproperty

tempdir = "/Users/vmoens/Downloads/Pong/1"


class AtariDQNExperienceReplay(TensorDictReplayBuffer):
    @_classproperty
    def available_datasets(cls):
        games = [
            "AirRaid",
            "Alien",
            "Amidar",
            "Assault",
            "Asterix",
            "Asteroids",
            "Atlantis",
            "BankHeist",
            "BattleZone",
            "BeamRider",
            "Berzerk",
            "Bowling",
            "Boxing",
            "Breakout",
            "Carnival",
            "Centipede",
            "ChopperCommand",
            "CrazyClimber",
            "DemonAttack",
            "DoubleDunk",
            "ElevatorAction",
            "Enduro",
            "FishingDerby",
            "Freeway",
            "Frostbite",
            "Gopher",
            "Gravitar",
            "Hero",
            "IceHockey",
            "Jamesbond",
            "JourneyEscape",
            "Kangaroo",
            "Krull",
            "KungFuMaster",
            "MontezumaRevenge",
            "MsPacman",
            "NameThisGame",
            "Phoenix",
            "Pitfall",
            "Pong",
            "Pooyan",
            "PrivateEye",
            "Qbert",
            "Riverraid",
            "RoadRunner",
            "Robotank",
            "Seaquest",
            "Skiing",
            "Solaris",
            "SpaceInvaders",
        ]
        return ["/".join((game, str(loop))) for game in games for loop in range(1, 6)]

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        from torchrl.data.datasets.utils import _get_root_dir
        self.root = Path(_get_root_dir("atari"))
        self._download_and_preproc()
        storage = _AtariStorage(self._root)
        super().__init__(storage=storage, collate_fn=lambda x: x)

    @property
    def root(self):
        return self._root
    @root.setter
    def root(self, value):
        self._root = Path(value)
    @property
    def dataset_path(self):
        return self._root / self.dataset_id
    def _download_and_preproc(self):
        if os.path.exists(self.dataset_path):
            # TODO: better check
            return
        with tempfile.TemporaryDirectory() as tempdir:
            command = f"gsutil -m cp -R gs://atari-replay-datasets/dqn/{self.dataset_id} {tempdir}"
            subprocess.run(command, shell=True)  #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gz_files = self._list_episodes(tempdir)
            for episode in gz_files:
                try:
                    path = self._root / str(episode)
                    self._preproc_episode(path, gz_files, episode)
                except Exception:
                    shutil.rmtree(path)
                    raise

    def _preproc_episode(self, path, gz_files, episode):
        print("preproc", episode)
        files = gz_files[episode]
        td = TensorDict({}, [])
        path = Path(path)
        for file in files:
            name = str(Path(file).parts[-1]).split(".")[0]
            with gzip.GzipFile(file, mode="rb") as f:
                file_content = f.read()
                file_content = io.BytesIO(file_content)
                file_content = np.load(file_content)
                t = torch.as_tensor(file_content)
            # Create the memmap file
            key = self._process_name(name)
            if key == ("data", "observation"):
                shape = t.shape
                shape = [shape[0] + 1] + list(shape[1:])
                filename = path / "data" / "observation.memmap"
                os.makedirs(filename.parent, exist_ok=True)
                mmap = MemoryMappedTensor.empty(shape, dtype=t.dtype, filename=filename)
                print('copying')
                mmap[:-1].copy_(t)
                td[key] = mmap
                # td["data", "next", key[1:]] = mmap[1:]
            else:
                if key in (("data", "reward"), ("data", "done"), ("data", "terminated")):
                    filename = path / "data" / "next" / (key[-1] + ".memmap")
                    os.makedirs(filename.parent, exist_ok=True)
                    mmap = MemoryMappedTensor.from_tensor(t, filename=filename)
                    td["data", "next", key[1:]] = mmap
                else:
                    filename = path
                    for i, _key in enumerate(key):
                        if i == len(key) - 1:
                            _key = _key + ".memmap"
                        filename = filename / _key
                    os.makedirs(filename.parent, exist_ok=True)
                    mmap = MemoryMappedTensor.from_tensor(t, filename=filename)
                    td[key] = mmap
        td.set_non_tensor("info", {"episode": episode, "path": path})
        td.memmap_(path, copy_existing=False)

    @staticmethod
    def _process_name(name):
        if name.endswith("_ckpt"):
            name = name[:-5]
        if "store" in name:
            key = ("data", name.split("_")[1])
        else:
            key = (name,)
        if key[-1] == "terminal":
            key = (*key[:-1], "terminated")
        return key

    def _list_episodes(self, download_path):
        path = download_path
        gz_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".gz"):
                    gz_files.append(os.path.join(root, file))
        episodes = defaultdict(list)
        for file in gz_files:
            filename = Path(file).parts[-1]
            name, episode, extension = str(filename).split(".")
            episode = int(episode)
            episodes[episode].append(file)
        return dict(sorted(episodes.items(), key=lambda x: x[0]))


class _AtariStorage(Storage):
    def __init__(self, path):
        self.path = Path(path)

    def __len__(self):
        return len(self.gz_files)

    def _get_episode(self, episode: int):
        path = self.path / str(episode)
        if os.path.exists(path):
            return self._load_episode(path)
        else:
            raise RuntimeError

    def _load_episode(self, path):
        return self._proc_td(TensorDict.load_memmap(path))

    def _proc_td(self, td):
        with td.unlock_():
            td["data", "next", "observation"] = td["data", "observation"][1:]
            td["data", "observation"] = td["data", "observation"][:-1]
            non_tensor =  td.exclude("data").to_dict()
            td = td["data"]
            td.auto_batch_size_(1)
            td.set_non_tensor("metadata", non_tensor)
            return td


    def get(self, index):
        if isinstance(index, int):
            return self._get_episode(index)
        if isinstance(index, tuple):
            if len(index) == 1:
                return self.get(index[0])
            return self.get(index[0])[..., index[1:]]
        if isinstance(index, torch.Tensor):
            if index.ndim == 0:
                return self[int(index)]
            if index.ndim > 1:
                raise RuntimeError("Only 1d tensors are accepted")
            # with ThreadPoolExecutor(16) as pool:
            results = map(self.__getitem__, index.tolist())
            return torch.stack(list(results))
        if isinstance(index, (range, list)):
            return self[torch.tensor(index)]
        return self[torch.arange(len(self))[index]]

import logging
logging.getLogger().setLevel(logging.INFO)
t0 = time.time()
print(AtariDQNExperienceReplay("Pong/5")[:3])
time.time() - t0
