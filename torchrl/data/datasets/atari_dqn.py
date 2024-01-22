# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

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

    tmpdir = "/Users/vmoens/.cache/atari_root"
    max_ep = 3

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        from torchrl.data.datasets.utils import _get_root_dir
        self.root = Path(_get_root_dir("atari"))
        self._download_and_preproc()
        storage = _AtariStorage(self.dataset_path)
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
            if self.tmpdir is not None:
                tempdir = self.tmpdir
            try:
                shutil.rmtree(tempdir)
                os.makedirs(tempdir, exist_ok=True)
            except:
                os.makedirs(tempdir, exist_ok=True)
            if not os.listdir(tempdir):
                os.makedirs(tempdir, exist_ok=True)
                # get the list of episodes
                command = f"gsutil -m ls -R gs://atari-replay-datasets/dqn/Pong/1/replay_logs"
                output = subprocess.run(
                    command,
                    shell=True, capture_output=True
                )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                files = [file.decode("utf-8").replace('$', '\$') for file in output.stdout.splitlines() if
                         file.endswith(b'.gz')]
                self.remote_gz_files = self._list_episodes(None, files)
                for episode, episode_files in self.remote_gz_files.items():
                    self._download_and_proc_episode(episode, episode_files, tempdir, self.dataset_path)

    @classmethod
    def _download_and_proc_episode(cls, episode, episode_files, tempdir, dataset_path):
        if episode >= 3:
            return
        tempdir = Path(tempdir)
        os.makedirs(tempdir/str(episode))
        files_str = ' '.join(episode_files)  # .decode("utf-8")
        print("downloading", files_str)
        command = f"gsutil -m cp {files_str} {tempdir}/{episode}"
        subprocess.run(
            command,
            shell=True
        )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        local_gz_files = cls._list_episodes(tempdir/str(episode))
        # we iterate over the dict but this one has length 1
        for episode in local_gz_files:
            path = dataset_path / str(episode)
            try:
                cls._preproc_episode(path, local_gz_files, episode)
            except Exception:
                shutil.rmtree(path)
                raise
        shutil.rmtree(tempdir / str(episode))

    @classmethod
    def _preproc_episode(cls, path, gz_files, episode):
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
            key = cls._process_name(name)
            if key == ("data", "observation"):
                shape = t.shape
                shape = [shape[0] + 1] + list(shape[1:])
                filename = path / "data" / "observation.memmap"
                os.makedirs(filename.parent, exist_ok=True)
                mmap = MemoryMappedTensor.empty(shape, dtype=t.dtype, filename=filename)
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

    @classmethod
    def _list_episodes(cls, download_path, gz_files=None):
        path = download_path
        if gz_files is None:
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

        def get_folders(path):
            return [name for name in os.listdir(path) if
                    os.path.isdir(os.path.join(path, name))]

        # Usage
        self.episodes = []
        folders = get_folders(path)
        for folder in folders:
            self.episodes.append(int(Path(folder).parts[-1]))
        self.frames_per_ep = 1000000
        self._episode_tds = []
        for episode in self.episodes:
            path = self.path / str(episode)
            self._episode_tds.append(self._load_episode(path))

    def __len__(self):
        return len(self.episodes) * self.frames_per_ep

    def _get_episode(self, item: int | torch.Tensor):
        # print('get episode', item)
        episode = item // self.frames_per_ep
        item = item % self.frames_per_ep
        if isinstance(item, int):
            unique_episodes = (episode,)
            episode_inverse = None
        else:
            unique_episodes, episode_inverse = torch.unique(episode, return_inverse=True)
        # print('unique_episodes, episode_inverse', unique_episodes, episode_inverse)
        out = []
        for i, episode in enumerate(unique_episodes):
            episode = int(episode)
            _item = item[episode_inverse == i] if episode_inverse is not None else item
            # print('_item', _item)
            path = self.path / str(episode)
            if os.path.exists(path):
                out.append( self._proc_td(self._episode_tds[episode], _item))
            else:
                raise RuntimeError
        # print('out', out)
        return torch.cat(out, 0)

    def _load_episode(self, path):
        return TensorDict.load_memmap(path)

    def _proc_td(self, td, index):
        obs_ = td["data", "observation"][index + 1]
        done = td["data", "next", "terminated"][index].bool()
        if done.ndim and done.any():
            obs_ = torch.masked_fill(obs_, done, 0)
        td_idx = td.empty()
        td_idx["next", "observation"] = obs_
        non_tensor = td.exclude("data").to_dict()
        td_idx.update(td["data"].apply(lambda x: x[index]))
        td_idx.auto_batch_size_(1)
        td_idx.set_non_tensor("metadata", non_tensor)
        return td_idx

    def get(self, index):
        if isinstance(index, int):
            return self._get_episode(index)
        if isinstance(index, tuple):
            if len(index) == 1:
                return self.get(index[0])
            return self.get(index[0])[(Ellipsis, *index[1:])]
        if isinstance(index, torch.Tensor):
            if index.ndim <= 1:
                return self._get_episode(index)
            else:
                raise RuntimeError("Only 1d tensors are accepted")
            # with ThreadPoolExecutor(16) as pool:
            # results = map(self.__getitem__, index.tolist())
            # return torch.stack(list(results))
        if isinstance(index, (range, list)):
            return self[torch.tensor(index)]
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self)
            step = index.step if index.step is not None else 1
            return self.get(torch.arange(start, stop, step))
        return self[torch.arange(len(self))[index]]

if __name__ == '__main__':
    # command = f"gsutil -m ls -R gs://atari-replay-datasets/dqn/Pong/1/replay_logs"
    # output = subprocess.run(
    #     command,
    #     shell=True, capture_output=True
    #     )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # print(output.stdout.splitlines(), type(output.stdout))
    # files = [file for file in output.stdout.splitlines() if file.endswith(b'.gz') and int(file.split(b'.')[-2]) <= 3]
    # files_str = b' '.join(files)
    # command = f"gsutil -m cp -R {files_str} {tempdir}"
    # subprocess.run(
    #     command,
    #     shell=True
    #     )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import logging
    logging.getLogger().setLevel(logging.INFO)
    dataset = AtariDQNExperienceReplay("Pong/5")
    # t0 = time.time()
    for _ in range(200):
        dataset[slice(0, 3000000, 50000)]
    # print(time.time() - t0)
