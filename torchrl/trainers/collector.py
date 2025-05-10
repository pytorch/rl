# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Create the `CollectorDataset` needed by the `pl.Trainer`."""

import typing as ty

import torch
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModule  # type: ignore

from torch.utils.data import IterableDataset
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase

from .accelerators import find_device


class CollectorDataset(IterableDataset):
    """Iterable Dataset containing the `ReplayBuffer` which will be
    updated with new experiences during training, and the `SyncDataCollector`."""

    def __init__(
        self,
        collector: SyncDataCollector | None = None,
        env: EnvBase | None = None,
        policy_module: TensorDictModule = None,
        frames_per_batch: int = 1,
        total_frames: int = 1000,
        device: torch.device = find_device(),
        split_trajs: bool = False,
        batch_size: int = 1,
        init_random_frames: int = 1,
    ) -> None:
        # Attributes
        self.batch_size = batch_size
        self.device = device
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        # Collector
        if collector is None:
            if env is None:
                raise ValueError(
                    "Please provide an environment when not providing a collector."
                )
            self.collector = SyncDataCollector(
                env,
                policy_module,
                frames_per_batch=self.frames_per_batch,
                total_frames=self.total_frames,
                device=self.device,
                storing_device=self.device,
                split_trajs=split_trajs,
                init_random_frames=init_random_frames,
            )
        else:
            self.collector = collector
        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.batch_size,
        )
        # States
        self.length: int | None = None

    def __iter__(self) -> ty.Iterator[TensorDict]:
        """Yield experiences from `SyncDataCollector` and store them in `ReplayBuffer`."""
        i = 0
        for i, tensordict_data in enumerate(self.collector):
            if not isinstance(tensordict_data, TensorDict):
                raise TypeError(
                    f"Collector returned an object of type {type(tensordict_data)}, expected {TensorDict}."
                )
            data_view: TensorDict = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view.cpu())
            yield tensordict_data.to(self.device)
        self.length = i

    def sample(self, **kwargs: ty.Any) -> TensorDict:
        """Sample from `ReplayBuffer`."""
        data: TensorDict = self.replay_buffer.sample(**kwargs)
        return data.to(self.device)
