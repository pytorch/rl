# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import gzip
import io
import json
import pathlib
import shutil

import mmap
import os
import subprocess
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import tqdm
from tensordict import NonTensorData, TensorDict, MemoryMappedTensor
from tensordict.utils import expand_right

from torchrl.data import LazyMemmapStorage, Storage, TensorDictReplayBuffer
from torchrl.envs.utils import _classproperty
from torch import multiprocessing as mp

class AtariDQNExperienceReplay(TensorDictReplayBuffer):
    """Atari DQN Experience replay class.

    The Atari DQN dataset (https://offline-rl.github.io/) is a collection of 5 training
    iterations of DQN over each of the Arari 2600 games for a total of 200 million frames.
    The sub-sampling rate (frame-skip) is equal to 4, meaning that each game dataset
    has 50 million steps in total.

    The data format follows the TED convention. Since the dataset is quite heavy,
    the data formatting is done on-line, at sampling time.

    To make training more modular, we split the dataset in each of the Atari games
    and separate each training round. Consequently, each dataset is presented as
    a Storage of length 50x10^6 elements. Under the hood, this dataset is split
    in 50 memory-mapped tensordicts of length 1 million each.

    Args:
        dataset_id (str): The dataset to be downloaded.
            Must be part of ``AtariDQNExperienceReplay.available_datasets``.
        batch_size (int): Batch-size used during sampling.
            Can be overridden by `data.sample(batch_size)` if necessary.

    Keyword Args:
        root (Path or str, optional): The AtariDQN dataset root directory.
            The actual dataset memory-mapped files will be saved under
            `<root>/<dataset_id>`. If none is provided, it defaults to
            ``~/.cache/torchrl/atari`.
        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as "force",
            in which case the downloaded data will be overwritten.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.writers.ImmutableDatasetWriter` will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs. Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :class:`~torchrl.envs.transforms.transforms.Compose` class.

    Examples:
        >>> from torchrl.data.datasets import AtariDQNExperienceReplay
        >>> from torchrl.data.replay_buffers import SliceSampler
        >>> sampler = SliceSampler()
        >>> dataset = AtariDQNExperienceReplay("Pong/5", batch_size=128, sampler=sampler)
        >>> for data in dataset:
        ...     print(data)
        ...     break

    As always, datasets should be composed using :class:`~torchrl.data.replay_buffers.ReplayBufferEnsemble`

    """
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

    # use _max_episodes for debugging, avoids downloading the entire dataset
    _max_episodes = 4

    def __init__(self, dataset_id:str, batch_size:int|None=None, *, root: str | Path | None = None, download: bool|str=True, sampler=None, writer=None, transform: "Transform" | None=None, num_procs: int=0, **kwargs):
        if dataset_id not in self.available_datasets:
            raise ValueError("The dataseet_id is not part of the available datasets. The dataset should be named <game_name>/<run> "
                             "where <game_name> is one of the Atari 2600 games and the run is a number betweeen 1 and 5. "
                             "The full list of accepted dataset_ids is available under AtariDQNExperienceReplay.available_datasets.")
        self.dataset_id = dataset_id
        from torchrl.data.datasets.utils import _get_root_dir
        if root is None:
            root = _get_root_dir("atari")
        self.root = root
        self.num_procs = num_procs
        if download == "force" or (download and not self._is_downloaded):
            try:
                self._download_and_preproc()
            except Exception:
                # remove temporary data
                if os.path.exists(self.dataset_path):
                    shutil.rmtree(self.dataset_path)
                raise
        storage = _AtariStorage(self.dataset_path)
        if writer is None:
            writer = ImmutableDatasetWriter()
        super().__init__(storage=storage, batch_size=batch_size, writer=writer, sampler=sampler, collate_fn=lambda x: x, transform=transform, **kwargs)

    @property
    def root(self)->Path:
        return self._root
    @root.setter
    def root(self, value):
        self._root = Path(value)
    @property
    def dataset_path(self) -> Path:
        return self._root / self.dataset_id
    @property
    def _is_downloaded(self):
        if os.path.exists(self.dataset_path / "processed.json"):
            with open(self.dataset_path / "processed.json", "r") as jsonfile:
                return json.load(jsonfile).get("processed", False)
        return False

    def _download_and_preproc(self):
        logging.info(f"Downloading and preprocessing dataset {self.dataset_id} with {self.num_procs} processes. This may take a while...")
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
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
                total_episodes = list(self.remote_gz_files)[-1]
                if self.num_procs == 0:
                    for episode, episode_files in self.remote_gz_files.items():
                        self._download_and_proc_episode(episode, episode_files, tempdir=tempdir, dataset_path=self.dataset_path, total_episodes=total_episodes)
                else:
                    func = functools.partial(self._download_and_proc_episode, tempdir=tempdir, dataset_path=self.dataset_path, total_episodes=total_episodes)
                    args = [(episode, episode_files) for (episode, episode_files) in self.remote_gz_files.items()]
                    with mp.Pool(self.num_procs) as pool:
                        pool.starmap(func, args)
        with open(self.dataset_path / "processed.json", "w") as file:
            json.dump({"processed": True}, file)

    @classmethod
    def _download_and_proc_episode(cls, episode, episode_files, *, tempdir, dataset_path, total_episodes):
        if cls._max_episodes is not None and episode >= cls._max_episodes:
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
        print(f'Concluded episode {episode} out of {total_episodes}')

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
        self._episode_tds = []
        frames_per_ep = {}
        for episode in self.episodes:
            path = self.path / str(episode)
            self._episode_tds.append(self._load_episode(path))
            # take away 1 because we padded with 1 empty val
            frames_per_ep[episode] = self._episode_tds[-1].get(("data", "observation")).shape[0] - 1

        frames_per_ep = torch.tensor([[episode, length] for (episode, length) in frames_per_ep.items()])
        frames_per_ep[:, 1] = frames_per_ep[:, 1].cumsum(0)
        self.frames_per_ep = torch.cat([torch.tensor([[-1, 0]]), frames_per_ep], 0)

    def __len__(self):
        return self.frames_per_ep[-1, 1].item()

    def _read_from_episodes(self, item: int | torch.Tensor):
        # We need to allocate each item to its storage.
        # We don't assume each storage has the same size (too expensive to test)
        # so we keep a map of each storage cumulative length and retrieve the
        # storages one after the other.
        episode = (item < self.frames_per_ep[1:, 1].unsqueeze(1)) & (item >= self.frames_per_ep[:-1, 1].unsqueeze(1))
        episode = episode.squeeze().nonzero()[:, 0]
        episode = self.frames_per_ep[episode+1, 0]
        item = item - self.frames_per_ep[episode, 1]
        if isinstance(item, int):
            unique_episodes = (episode,)
            episode_inverse = None
        else:
            unique_episodes, episode_inverse = torch.unique(episode, return_inverse=True)
            unique_episodes = unique_episodes.tolist()
        out = []
        for i, episode in enumerate(unique_episodes):
            _item = item[episode_inverse == i] if episode_inverse is not None else item
            out.append( self._proc_td(self._episode_tds[episode], _item))
        return torch.cat(out, 0)

    def _load_episode(self, path):
        return TensorDict.load_memmap(path)

    def _proc_td(self, td, index):
        td_data = td.get("data")
        obs_ = td_data.get(("observation"))[index + 1]
        done = td_data.get(("next", "terminated"))[index].squeeze(-1).bool()
        if done.ndim and done.any():
            obs_ = torch.index_fill(obs_, 0, done.nonzero().squeeze(), 0)
            # obs_ = torch.masked_fill(obs_, done, 0)
            # obs_ = torch.masked_fill(obs_, expand_right(done, obs_.shape), 0)
            # obs_ = torch.where(~expand_right(done, obs_.shape), obs_, 0)
        td_idx = td.empty()
        td_idx.set(("next", "observation"), obs_)
        non_tensor = td.exclude("data").to_dict()
        td_idx.update(td_data.apply(lambda x: x[index]))
        if isinstance(index, torch.Tensor):
            td_idx.batch_size = [len(index)]
        td_idx.set_non_tensor("metadata", non_tensor)
        return td_idx

    def get(self, index):
        if isinstance(index, int):
            return self._read_from_episodes(index)
        if isinstance(index, tuple):
            if len(index) == 1:
                return self.get(index[0])
            return self.get(index[0])[(Ellipsis, *index[1:])]
        if isinstance(index, torch.Tensor):
            if index.ndim <= 1:
                return self._read_from_episodes(index)
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
    import logging
    logging.getLogger().setLevel(logging.INFO)
    dataset = AtariDQNExperienceReplay("Pong/5", num_procs=4)
    for _ in range(100):
        out = dataset[slice(0, 3000000, 10000)]