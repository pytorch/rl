# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import tempfile

from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer, Storage
import os
import gzip
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tensordict import TensorDict, NonTensorData
import torch
import io
import mmap
from pathlib import Path
from collections import defaultdict

tempdir = "/Users/vmoens/Downloads/Pong/1"


class AtariDQNExperienceReplay(TensorDictReplayBuffer):
    available_datasets = ["Pong/1", ]

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        storage = _AtariStorage(tempdir)
        super().__init__(storage=storage, collate_fn=lambda x: x)


class _AtariStorage(Storage):
    def __init__(self, path):
        self.path = path
        self.gz_files = self._list_episodes(self.path)

    def __len__(self):
        return len(self.gz_files)

    def _get_episode(self, episode):
        gz_files = self.gz_files
        files = gz_files[episode]
        td = {}
        for file in files:
            name = str(Path(file).parts[-1]).split(".")[0]
            with gzip.GzipFile(file, mode="rb") as f:
                t0 = time.time()
                file_content = f.read()
                t1 = time.time()
                file_content = io.BytesIO(file_content)
                t2 = time.time()
                file_content = np.load(file_content)
                t3 = time.time()
                print(t1 - t0, t2 - t1, t3 - t2)
                t = torch.as_tensor(file_content)
            td[self._process_name(name)] = t
        td = TensorDict.from_dict(td)
        td = td["data"].set(
            "metadata",
            NonTensorData(
                td.exclude("data").to_dict(),
                batch_size=td["data"].batch_size
                )
            )
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

    @staticmethod
    def _process_name(name):
        if "store" in name:
            return ("data", name.split("_")[1])
        if name.endswith("_ckpt"):
            return name[:-5]

    def _list_episodes(self, path):
        gz_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.gz'):
                    gz_files.append(os.path.join(root, file))
        episodes = defaultdict(list)
        for file in gz_files:
            filename = Path(file).parts[-1]
            name, episode, extension = str(filename).split(".")
            episode = int(episode)
            episodes[episode].append(file)
        return episodes

t0 = time.time()
AtariDQNExperienceReplay(AtariDQNExperienceReplay.available_datasets[0])[:3]
time.time()-t0
