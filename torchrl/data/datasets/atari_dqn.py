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
import torch
import io
from pathlib import Path

class AtariDQNExperienceReplay(TensorDictReplayBuffer):
    available_datasets = ["Pong/1",]
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        storage = LazyMemmapStorage(1_000_000)
        super().__init__(storage=storage)


    def _download_dataset(self):
        # with tempfile.TemporaryDirectory() as tempdir:
        tempdir = "/Users/vmoens/Downloads/Pong/1"
        # command = f"gsutil -m cp -R gs://atari-replay-datasets/dqn/{self.dataset_id} {tempdir}"
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        gz_files = []
        for root, dirs, files in os.walk(tempdir):
            for file in files:
                if file.endswith('.gz'):
                    gz_files.append(os.path.join(root, file))

        def _count_files(pattern):
            return sum(pattern in filename for filename in gz_files)

        pbar = tqdm.tqdm(gz_files)
        for file in pbar:
            name = str(Path(file).parts[-1]).split(".")[0]
            # with open(file, "r") as fopen:
            if "obs" in file:
                print(name, file)
                print("count", _count_files(name))
                with gzip.GzipFile(file) as f:
                    file_content = f.read()
                t = torch.as_tensor(np.load(io.BytesIO(file_content)))
                print(t.shape, t.dtype)
                break

AtariDQNExperienceReplay(AtariDQNExperienceReplay.available_datasets[0])._download_dataset()
