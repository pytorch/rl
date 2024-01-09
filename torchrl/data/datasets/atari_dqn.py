# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import tempfile

from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
import os
import gzip
import tqdm
import numpy as np
from tensordict import TensorDict
import torch
import io
from pathlib import Path
from collections import defaultdict

tempdir = "/Users/vmoens/Downloads/Pong/1"


class AtariDQNExperienceReplay(TensorDictReplayBuffer):
    available_datasets = ["Pong/1", ]

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        storage = LazyMemmapStorage(1_000_000)
        super().__init__(storage=storage)

    # def _download_dataset(self):
    #     # with tempfile.TemporaryDirectory() as tempdir:
    #     # command = f"gsutil -m cp -R gs://atari-replay-datasets/dqn/{self.dataset_id} {tempdir}"
    #     # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _get_episode(self, episode, path):
        gz_files = self._list_episodes(path)
        files = gz_files[episode]
        td = {}
        for file in files:
            name = str(Path(file).parts[-1]).split(".")[0]
            with gzip.GzipFile(file) as f:
                file_content = f.read()
            t = torch.as_tensor(np.load(io.BytesIO(file_content)))
            td[self._process_name(name)] = t
        td = TensorDict.from_dict(td)
        return td

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


AtariDQNExperienceReplay(
    AtariDQNExperienceReplay.available_datasets[0]
    )._get_episode(0, tempdir)
