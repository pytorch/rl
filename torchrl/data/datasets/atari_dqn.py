# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import gzip
import io
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tensordict import MemoryMappedTensor, TensorDict, TensorDictBase
from torch import multiprocessing as mp
from torchrl._utils import logger as torchrl_logger
from torchrl.data.datasets.common import BaseDatasetExperienceReplay
from torchrl.data.replay_buffers.samplers import (
    SamplerWithoutReplacement,
    SliceSampler,
    SliceSamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import Storage, TensorStorage
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs.utils import _classproperty


class AtariDQNExperienceReplay(BaseDatasetExperienceReplay):
    """Atari DQN Experience replay class.

    The Atari DQN dataset (https://offline-rl.github.io/) is a collection of 5 training
    iterations of DQN over each of the Arari 2600 games for a total of 200 million frames.
    The sub-sampling rate (frame-skip) is equal to 4, meaning that each game dataset
    has 50 million steps in total.

    The data format follows the :ref:`TED convention <TED-format>`. Since the dataset is quite heavy,
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
            `~/.cache/torchrl/atari`.atari`.
        num_procs (int, optional): number of processes to launch for preprocessing.
            Has no effect whenever the data is already downloaded. Defaults to 0
            (no multiprocessing used).
        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as ``"force"``,
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
        num_slices (int, optional): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``. Defaults to ``None`` (no slice sampling).
            The ``sampler`` arg will override this value.
        slice_len (int, optional): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``. Defaults to ``None`` (no slice sampling).
            The ``sampler`` arg will override this value.
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`torchrl.collectors.split_trajectories`. Defaults to ``True``.
            The ``sampler`` arg will override this value.
        replacement (bool, optional): if ``False``, sampling will occur without replacement.
            The ``sampler`` arg will override this value.
        mp_start_method (str, optional): the start method for multiprocessed
            download. Defaults to ``"fork"``.

    Attributes:
        available_datasets: list of available datasets, formatted as `<game_name>/<run>`. Example:
            `"Pong/5"`, `"Krull/2"`, ...
        dataset_id (str): the name of the dataset.
        episodes (torch.Tensor): a 1d tensor indicating to what run each of the
            1M frames belongs. To be used with :class:`~torchrl.data.replay_buffers.SliceSampler`
            to cheaply sample slices of episodes.

    Examples:
        >>> from torchrl.data.datasets import AtariDQNExperienceReplay
        >>> dataset = AtariDQNExperienceReplay("Pong/5", batch_size=128)
        >>> for data in dataset:
        ...     print(data)
        ...     break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
                index: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int64, is_shared=False),
                metadata: NonTensorData(
                    data={'invalid_range': MemoryMappedTensor([999998, 999999,      0,      1,      2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Pong/5'}},
                    batch_size=torch.Size([128]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
                        observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
                        truncated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False)},
                    batch_size=torch.Size([128]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
                truncated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([128]),
            device=None,
            is_shared=False)

    .. warning::
      Atari-DQN does not provide the next observation after a termination signal.
      In other words, there is no way to obtain the ``("next", "observation")`` state
      when ``("next", "done")`` is ``True``. This value is filled with 0s but should
      not be used in practice. If TorchRL's value estimators (:class:`~torchrl.objectives.values.ValueEstimator`)
      are used, this should not be an issue.

    .. note::
      Because the construction of the sampler for episode sampling is slightly
      convoluted, we made it easy for users to pass the arguments of the
      :class:`~torchrl.data.replay_buffers.SliceSampler` directly to the
      ``AtariDQNExperienceReplay`` dataset: any of the ``num_slices`` or
      ``slice_len`` arguments will make the sampler an instance of
      :class:`~torchrl.data.replay_buffers.SliceSampler`. The ``strict_length``
      can also be passed.

        >>> from torchrl.data.datasets import AtariDQNExperienceReplay
        >>> from torchrl.data.replay_buffers import SliceSampler
        >>> dataset = AtariDQNExperienceReplay("Pong/5", batch_size=128, slice_len=64)
        >>> for data in dataset:
        ...     print(data)
        ...     print(data.get("index"))  # indices are in 4 groups of consecutive values
        ...     break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
                index: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int64, is_shared=False),
                metadata: NonTensorData(
                    data={'invalid_range': MemoryMappedTensor([999998, 999999,      0,      1,      2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Pong/5'}},
                    batch_size=torch.Size([128]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([128]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
                truncated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([128]),
            device=None,
            is_shared=False)
        tensor([2657628, 2657629, 2657630, 2657631, 2657632, 2657633, 2657634, 2657635,
                2657636, 2657637, 2657638, 2657639, 2657640, 2657641, 2657642, 2657643,
                2657644, 2657645, 2657646, 2657647, 2657648, 2657649, 2657650, 2657651,
                2657652, 2657653, 2657654, 2657655, 2657656, 2657657, 2657658, 2657659,
                2657660, 2657661, 2657662, 2657663, 2657664, 2657665, 2657666, 2657667,
                2657668, 2657669, 2657670, 2657671, 2657672, 2657673, 2657674, 2657675,
                2657676, 2657677, 2657678, 2657679, 2657680, 2657681, 2657682, 2657683,
                2657684, 2657685, 2657686, 2657687, 2657688, 2657689, 2657690, 2657691,
                1995687, 1995688, 1995689, 1995690, 1995691, 1995692, 1995693, 1995694,
                1995695, 1995696, 1995697, 1995698, 1995699, 1995700, 1995701, 1995702,
                1995703, 1995704, 1995705, 1995706, 1995707, 1995708, 1995709, 1995710,
                1995711, 1995712, 1995713, 1995714, 1995715, 1995716, 1995717, 1995718,
                1995719, 1995720, 1995721, 1995722, 1995723, 1995724, 1995725, 1995726,
                1995727, 1995728, 1995729, 1995730, 1995731, 1995732, 1995733, 1995734,
                1995735, 1995736, 1995737, 1995738, 1995739, 1995740, 1995741, 1995742,
                1995743, 1995744, 1995745, 1995746, 1995747, 1995748, 1995749, 1995750])

    .. note::
      As always, datasets should be composed using :class:`~torchrl.data.replay_buffers.ReplayBufferEnsemble`:

        >>> from torchrl.data.datasets import AtariDQNExperienceReplay
        >>> from torchrl.data.replay_buffers import ReplayBufferEnsemble
        >>> # we change this parameter for quick experimentation, in practice it should be left untouched
        >>> AtariDQNExperienceReplay._max_runs = 2
        >>> dataset_asterix = AtariDQNExperienceReplay("Asterix/5", batch_size=128, slice_len=64, num_procs=4)
        >>> dataset_pong = AtariDQNExperienceReplay("Pong/5", batch_size=128, slice_len=64, num_procs=4)
        >>> dataset = ReplayBufferEnsemble(dataset_pong, dataset_asterix, batch_size=128, sample_from_all=True)
        >>> sample = dataset.sample()
        >>> print("first sample, Asterix", sample[0])
        first sample, Asterix TensorDict(
            fields={
                action: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
                index: TensorDict(
                    fields={
                        buffer_ids: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False),
                        index: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([64]),
                    device=None,
                    is_shared=False),
                metadata: NonTensorData(
                    data={'invalid_range': MemoryMappedTensor([999998, 999999,      0,      1,      2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Pong/5'},
                    batch_size=torch.Size([64]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([64]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
                truncated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([64]),
            device=None,
            is_shared=False)
        >>> print("second sample, Pong", sample[1])
        second sample, Pong TensorDict(
            fields={
                action: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
                index: TensorDict(
                    fields={
                        buffer_ids: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False),
                        index: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([64]),
                    device=None,
                    is_shared=False),
                metadata: NonTensorData(
                    data={'invalid_range': MemoryMappedTensor([999998, 999999,      0,      1,      2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Asterix/5'},
                    batch_size=torch.Size([64]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([64]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
                truncated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([64]),
            device=None,
            is_shared=False)
        >>> print("Aggregate (metadata hidden)", sample)
        Aggregate (metadata hidden) LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
                index: LazyStackedTensorDict(
                    fields={
                        buffer_ids: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int64, is_shared=False),
                        index: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int64, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([2, 64]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                metadata: LazyStackedTensorDict(
                    fields={
                    },
                    exclusive_fields={
                    },
                    batch_size=torch.Size([2, 64]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([2, 64]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                observation: Tensor(shape=torch.Size([2, 64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.uint8, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([2, 64]),
            device=None,
            is_shared=False,
            stack_dim=0)

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

    # If we want to keep track of the original atari files
    tmpdir = None
    # use _max_runs for debugging, avoids downloading the entire dataset
    _max_runs = None

    def __init__(
        self,
        dataset_id: str,
        batch_size: int | None = None,
        *,
        root: str | Path | None = None,
        download: bool | str = True,
        sampler=None,
        writer=None,
        transform: Transform | None = None,  # noqa: F821
        num_procs: int = 0,
        num_slices: int | None = None,
        slice_len: int | None = None,
        strict_len: bool = True,
        replacement: bool = True,
        mp_start_method: str = "fork",
        **kwargs,
    ):
        if dataset_id not in self.available_datasets:
            raise ValueError(
                "The dataseet_id is not part of the available datasets. The dataset should be named <game_name>/<run> "
                "where <game_name> is one of the Atari 2600 games and the run is a number between 1 and 5. "
                "The full list of accepted dataset_ids is available under AtariDQNExperienceReplay.available_datasets."
            )
        self.dataset_id = dataset_id
        from torchrl.data.datasets.utils import _get_root_dir

        if root is None:
            root = _get_root_dir("atari")
        self.root = root
        self.num_procs = num_procs
        self.mp_start_method = mp_start_method
        if download == "force" or (download and not self._is_downloaded):
            try:
                self._download_and_preproc()
            except Exception:
                # remove temporary data
                if os.path.exists(self.dataset_path):
                    shutil.rmtree(self.dataset_path)
                raise
        if self._downloaded_and_preproc:
            storage = TensorStorage(TensorDict.load_memmap(self.dataset_path))
        else:
            storage = _AtariStorage(self.dataset_path)
        if writer is None:
            writer = ImmutableDatasetWriter()
        if sampler is None:
            if num_slices is not None or slice_len is not None:
                if not replacement:
                    sampler = SliceSamplerWithoutReplacement(
                        num_slices=num_slices,
                        slice_len=slice_len,
                        trajectories=storage.episodes,
                    )
                else:
                    sampler = SliceSampler(
                        num_slices=num_slices,
                        slice_len=slice_len,
                        trajectories=storage.episodes,
                        cache_values=True,
                    )
            elif not replacement:
                sampler = SamplerWithoutReplacement()

        super().__init__(
            storage=storage,
            batch_size=batch_size,
            writer=writer,
            sampler=sampler,
            collate_fn=lambda x: x,
            transform=transform,
            **kwargs,
        )

    @property
    def episodes(self):
        return self._storage.episodes

    @property
    def root(self) -> Path:
        return self._root

    @root.setter
    def root(self, value):
        self._root = Path(value)

    @property
    def dataset_path(self) -> Path:
        return self._root / self.dataset_id

    @property
    def _downloaded_and_preproc(self):
        return os.path.exists(self.dataset_path / "meta.json")

    @property
    def _is_downloaded(self):
        if os.path.exists(self.dataset_path / "meta.json"):
            return True
        if os.path.exists(self.dataset_path / "processed.json"):
            with open(self.dataset_path / "processed.json") as jsonfile:
                return json.load(jsonfile).get("processed", False) == self._max_runs
        return False

    def _download_and_preproc(self):
        torchrl_logger.info(
            f"Downloading and preprocessing dataset {self.dataset_id} with {self.num_procs} processes. This may take a while..."
        )
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
        with tempfile.TemporaryDirectory() as tempdir:
            if self.tmpdir is not None:
                tempdir = self.tmpdir
            if not os.listdir(tempdir):
                os.makedirs(tempdir, exist_ok=True)
                # get the list of runs
                try:
                    subprocess.run(
                        ["gsutil", "version"], check=True, capture_output=True
                    )
                except subprocess.CalledProcessError:
                    raise RuntimeError("gsutil is not installed or not found in PATH.")
                command = f"gsutil -m ls -R gs://atari-replay-datasets/dqn/{self.dataset_id}/replay_logs"
                output = subprocess.run(
                    command, shell=True, capture_output=True
                )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                files = [
                    file.decode("utf-8").replace("$", r"\$")  # noqa: W605
                    for file in output.stdout.splitlines()
                    if file.endswith(b".gz")
                ]
                self.remote_gz_files = self._list_runs(None, files)
                remote_gz_files = list(self.remote_gz_files)
                if not len(remote_gz_files):
                    raise RuntimeError("No files in file list.")

                total_runs = remote_gz_files[-1]
                if self.num_procs == 0:
                    for run, run_files in self.remote_gz_files.items():
                        self._download_and_proc_split(
                            run,
                            run_files,
                            tempdir=tempdir,
                            dataset_path=self.dataset_path,
                            total_episodes=total_runs,
                            max_runs=self._max_runs,
                            multithreaded=True,
                        )
                else:
                    func = functools.partial(
                        self._download_and_proc_split,
                        tempdir=tempdir,
                        dataset_path=self.dataset_path,
                        total_episodes=total_runs,
                        max_runs=self._max_runs,
                        multithreaded=False,
                    )
                    args = [
                        (run, run_files)
                        for (run, run_files) in self.remote_gz_files.items()
                    ]
                    ctx = mp.get_context(self.mp_start_method)
                    with ctx.Pool(self.num_procs) as pool:
                        pool.starmap(func, args)
        with open(self.dataset_path / "processed.json", "w") as file:
            # we save self._max_runs such that changing the number of runs to process
            # forces the data to be re-downloaded
            json.dump({"processed": self._max_runs}, file)

    @classmethod
    def _download_and_proc_split(
        cls,
        run,
        run_files,
        *,
        tempdir,
        dataset_path,
        total_episodes,
        max_runs,
        multithreaded=True,
    ):
        if (max_runs is not None) and (run >= max_runs):
            return
        tempdir = Path(tempdir)
        os.makedirs(tempdir / str(run))
        files_str = " ".join(run_files)  # .decode("utf-8")
        torchrl_logger.info(f"downloading {files_str}")
        if multithreaded:
            command = f"gsutil -m cp {files_str} {tempdir}/{run}"
        else:
            command = f"gsutil cp {files_str} {tempdir}/{run}"
        subprocess.run(
            command, shell=True
        )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        local_gz_files = cls._list_runs(tempdir / str(run))
        # we iterate over the dict but this one has length 1
        for run in local_gz_files:
            path = dataset_path / str(run)
            try:
                cls._preproc_run(path, local_gz_files, run)
            except Exception:
                shutil.rmtree(path)
                raise
        shutil.rmtree(tempdir / str(run))
        torchrl_logger.info(f"Concluded run {run} out of {total_episodes}")

    @classmethod
    def _preproc_run(cls, path, gz_files, run):
        files = gz_files[run]
        td = TensorDict()
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
                if key in (
                    ("data", "reward"),
                    ("data", "done"),
                    ("data", "terminated"),
                ):
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
        td.set_non_tensor("dataset_id", "/".join(path.parts[-3:-1]))
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
    def _list_runs(cls, download_path, gz_files=None) -> dict:
        path = download_path
        if gz_files is None:
            gz_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".gz"):
                        gz_files.append(os.path.join(root, file))
        runs = defaultdict(list)
        for file in gz_files:
            filename = Path(file).parts[-1]
            name, episode, extension = str(filename).split(".")
            episode = int(episode)
            runs[episode].append(file)
        return dict(sorted(runs.items(), key=lambda x: x[0]))

    def preprocess(
        self,
        fn: Callable[[TensorDictBase], TensorDictBase],
        dim: int = 0,
        num_workers: int | None = None,
        *,
        chunksize: int | None = None,
        num_chunks: int | None = None,
        pool: mp.Pool | None = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: int | None = None,
        worker_threads: int = 1,
        index_with_generator: bool = False,
        pbar: bool = False,
        mp_start_method: str | None = None,
        dest: str | Path,
        num_frames: int | None = None,
    ):
        # Copy data to a tensordict
        with tempfile.TemporaryDirectory() as tmpdir:
            first_item = self[0]
            metadata = first_item.pop("metadata")

            mmap = fn(first_item)
            if num_frames is None:
                num_frames = len(self)
            mmap = mmap.expand(num_frames, *first_item.shape)
            mmap = mmap.memmap_like(tmpdir, num_threads=32)
            with mmap.unlock_():
                mmap["_indices"] = torch.arange(mmap.shape[0])
            mmap.memmap_(tmpdir, num_threads=32)

            def func(mmap: TensorDictBase):
                idx = mmap["_indices"]
                orig = self[idx].exclude("metadata")
                orig = fn(orig)
                mmap.update(orig, inplace=True)
                return

            if dim != 0:
                raise RuntimeError("dim != 0 is not supported.")

            mmap.map(
                fn=CloudpickleWrapper(func),
                dim=dim,
                num_workers=num_workers,
                chunksize=chunksize,
                num_chunks=num_chunks,
                pool=pool,
                generator=generator,
                max_tasks_per_child=max_tasks_per_child,
                worker_threads=worker_threads,
                index_with_generator=index_with_generator,
                mp_start_method=mp_start_method,
                pbar=pbar,
            )

            with mmap.unlock_():
                return TensorStorage(mmap.set("metadata", metadata))


class _AtariStorage(Storage):
    def __init__(self, path):
        self.path = Path(path)

        def get_folders(path):
            return [
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]

        # Usage
        self.splits = []
        folders = get_folders(path)
        for folder in folders:
            self.splits.append(int(Path(folder).parts[-1]))
        self.splits = sorted(self.splits)
        self._split_tds = []
        frames_per_split = {}
        for split in self.splits:
            path = self.path / str(split)
            self._split_tds.append(self._load_split(path))
            # take away 1 because we padded with 1 empty val
            frames_per_split[split] = (
                self._split_tds[-1].get(("data", "observation")).shape[0] - 1
            )

        frames_per_split = torch.tensor(
            [[split, length] for (split, length) in frames_per_split.items()]
        )
        frames_per_split[:, 1] = frames_per_split[:, 1].cumsum(0)
        self.frames_per_split = torch.cat(
            # [torch.tensor([[-1, 0]]), frames_per_split], 0
            [torch.tensor([[-1, 0]]), frames_per_split],
            0,
        )

        # retrieve episodes
        self.episodes = torch.cumsum(
            torch.cat(
                [td.get(("data", "next", "terminated")) for td in self._split_tds], 0
            ),
            0,
        )
        super().__init__(max_size=len(self))

    def __len__(self):
        return self.frames_per_split[-1, 1].item()

    def _read_from_splits(self, item: int | torch.Tensor):
        # We need to allocate each item to its storage.
        # We don't assume each storage has the same size (too expensive to test)
        # so we keep a map of each storage cumulative length and retrieve the
        # storages one after the other.
        item = torch.as_tensor(item)
        if not item.ndim:
            is_int = True
            item = item.reshape(-1)
        else:
            is_int = False
        split = (item < self.frames_per_split[1:, 1].unsqueeze(1)) & (
            item >= self.frames_per_split[:-1, 1].unsqueeze(1)
        )
        # split_tmp, idx = split.squeeze().nonzero().unbind(-1)
        split_tmp, idx = split.nonzero().unbind(-1)
        split = split_tmp.squeeze()
        idx = idx.squeeze()

        if not is_int:
            split = torch.zeros_like(split_tmp)
            split[idx] = split_tmp
        split = self.frames_per_split[split + 1, 0]
        item = item - self.frames_per_split[split, 1]
        if is_int:
            item = item.squeeze()
            return self._proc_td(self._split_tds[split], item)
        unique_splits, split_inverse = torch.unique(split, return_inverse=True)
        unique_splits = unique_splits.tolist()
        out = []
        for i, split in enumerate(unique_splits):
            _item = item[split_inverse == i] if split_inverse is not None else item
            out.append(self._proc_td(self._split_tds[split], _item))
        return torch.cat(out, 0)

    def _load_split(self, path):
        return TensorDict.load_memmap(path)

    def _proc_td(self, td, index):
        td_data = td.get("data")
        obs_ = td_data.get("observation")[index + 1]
        done = td_data.get(("next", "terminated"))[index].squeeze(-1).bool()
        if done.ndim and done.any():
            obs_ = torch.index_fill(obs_, 0, done.nonzero().squeeze(), 0)
        td_idx = td.empty()
        td_idx.set(("next", "observation"), obs_)
        non_tensor = td.exclude("data").to_dict()
        td_idx.update(td_data.apply(lambda x: x[index]))
        if isinstance(index, torch.Tensor) and index.ndim:
            td_idx.batch_size = [len(index)]
        td_idx.set_non_tensor("metadata", non_tensor)

        terminated = td_idx.get(("next", "terminated"))
        zterminated = torch.zeros_like(terminated)
        td_idx.set(("next", "done"), terminated.clone())
        td_idx.set(("next", "truncated"), zterminated)
        td_idx.set("terminated", zterminated)
        td_idx.set("done", zterminated)
        td_idx.set("truncated", zterminated)

        return td_idx

    def get(self, index):
        if isinstance(index, int):
            return self._read_from_splits(index)
        if isinstance(index, tuple):
            if len(index) == 1:
                return self.get(index[0])
            return self.get(index[0])[(Ellipsis, *index[1:])]
        if isinstance(index, torch.Tensor):
            if index.ndim <= 1:
                return self._read_from_splits(index)
            elif index.shape[1] == 1:
                index = index.squeeze(1)
                return self.get(index)
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
