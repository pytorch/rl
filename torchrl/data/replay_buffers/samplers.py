# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
import textwrap
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from tensordict import MemoryMappedTensor, TensorDict
from tensordict.utils import NestedKey

from torchrl._extension import EXTENSION_WARNING

from torchrl._utils import _replace_last
from torchrl.data.replay_buffers.storages import Storage, StorageEnsemble, TensorStorage

try:
    from torchrl._torchrl import (
        MinSegmentTreeFp32,
        MinSegmentTreeFp64,
        SumSegmentTreeFp32,
        SumSegmentTreeFp64,
    )
except ImportError:
    warnings.warn(EXTENSION_WARNING)

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


class Sampler(ABC):
    """A generic sampler base class for composable Replay Buffers."""

    @abstractmethod
    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        ...

    def add(self, index: int) -> None:
        return

    def extend(self, index: torch.Tensor) -> None:
        return

    def update_priority(
        self, index: Union[int, torch.Tensor], priority: Union[float, torch.Tensor]
    ) -> dict:
        return

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        return

    @property
    def default_priority(self) -> float:
        return 1.0

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @property
    def ran_out(self) -> bool:
        # by default, samplers never run out
        return False

    @abstractmethod
    def _empty(self):
        ...

    @abstractmethod
    def dumps(self, path):
        ...

    @abstractmethod
    def loads(self, path):
        ...


class RandomSampler(Sampler):
    """A uniformly random sampler for composable replay buffers.

    Args:
        batch_size (int, optional): if provided, the batch size to be used by
            the replay buffer when calling :meth:`~.ReplayBuffer.sample`.

    """

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        index = torch.randint(0, len(storage), (batch_size,))
        return index, {}

    def _empty(self):
        pass

    def dumps(self, path):
        # no op
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return


class SamplerWithoutReplacement(Sampler):
    """A data-consuming sampler that ensures that the same sample is not present in consecutive batches.

    Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept and (unlike with torch dataloaders)
            completed with other samples from a fresh indices permutation.
            Defaults to ``False``.
        shuffle (bool, optional): if ``False``, the items are not randomly
            permuted. This enables to iterate over the replay buffer in the
            order the data was collected. Defaults to ``True``.

    *Caution*: If the size of the storage changes in between two calls, the samples will be re-shuffled
    (as we can't generally keep track of which samples have been sampled before and which haven't).

    Similarly, it is expected that the storage content remains the same in between two calls,
    but this is not enforced.

    When the sampler reaches the end of the list of available indices, a new sample order
    will be generated and the resulting indices will be completed with this new draw, which
    can lead to duplicated indices, unless the :obj:`drop_last` argument is set to ``True``.

    """

    def __init__(self, drop_last: bool = False, shuffle: bool = True):
        self._sample_list = None
        self.len_storage = 0
        self.drop_last = drop_last
        self._ran_out = False
        self.shuffle = shuffle

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        with open(path / "sampler_metadata.json", "w") as file:
            json.dump(
                self.state_dict(),
                file,
            )

    def loads(self, path):
        with open(path / "sampler_metadata.json", "r") as file:
            metadata = json.load(file)
        self.load_state_dict(metadata)

    def _get_sample_list(self, storage: Storage, len_storage: int):
        if storage is None:
            device = self._sample_list.device
        else:
            device = storage.device if hasattr(storage, "device") else None
        if self.shuffle:
            self._sample_list = torch.randperm(len_storage, device=device)
        else:
            self._sample_list = torch.arange(len_storage, device=device)

    def _single_sample(self, len_storage, batch_size):
        index = self._sample_list[:batch_size]
        self._sample_list = self._sample_list[batch_size:]

        # check if we have enough elements for one more batch, assuming same batch size
        # will be used each time sample is called
        if self._sample_list.numel() == 0 or (
            self.drop_last and len(self._sample_list) < batch_size
        ):
            self.ran_out = True
            self._get_sample_list(storage=None, len_storage=len_storage)
        else:
            self.ran_out = False
        return index

    def _storage_len(self, storage):
        return len(storage)

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        len_storage = self._storage_len(storage)
        if len_storage == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        if not len_storage:
            raise RuntimeError("An empty storage was passed")
        if self.len_storage != len_storage or self._sample_list is None:
            self._get_sample_list(storage, len_storage)
        if len_storage < batch_size and self.drop_last:
            raise ValueError(
                f"The batch size ({batch_size}) is greater than the storage capacity ({len_storage}). "
                "This makes it impossible to return a sample without repeating indices. "
                "Consider changing the sampler class or turn the 'drop_last' argument to False."
            )
        self.len_storage = len_storage
        index = self._single_sample(len_storage, batch_size)
        # we 'always' return the indices. The 'drop_last' just instructs the
        # sampler to turn to 'ran_out = True` whenever the next sample
        # will be too short. This will be read by the replay buffer
        # as a signal for an early break of the __iter__().
        return index, {}

    @property
    def ran_out(self):
        return self._ran_out

    @ran_out.setter
    def ran_out(self, value):
        self._ran_out = value

    def _empty(self):
        self._sample_list = None
        self.len_storage = 0
        self._ran_out = False

    def state_dict(self) -> Dict[str, Any]:
        return OrderedDict(
            len_storage=self.len_storage,
            _sample_list=self._sample_list,
            drop_last=self.drop_last,
            _ran_out=self._ran_out,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.len_storage = state_dict["len_storage"]
        self._sample_list = state_dict["_sample_list"]
        self.drop_last = state_dict["drop_last"]
        self._ran_out = state_dict["_ran_out"]


class PrioritizedSampler(Sampler):
    """Prioritized sampler for replay buffer.

    Presented in "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay." (https://arxiv.org/abs/1511.05952)

    Args:
        max_capacity (int): maximum capacity of the buffer.
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float, optional): delta added to the priorities to ensure that the buffer
            does not contain null priorities. Defaults to 1e-8.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectory). Can be one of "max", "min",
            "median" or "mean".

    Examples:
        >>> from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, PrioritizedSampler
        >>> from tensordict import TensorDict
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(10), sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0))
        >>> priority = torch.tensor([0, 1000])
        >>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
        >>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
        >>> rb.add(data_0)
        >>> rb.add(data_1)
        >>> rb.update_priority(torch.tensor([0, 1]), priority=priority)
        >>> sample, info = rb.sample(10, return_info=True)
        >>> print(sample)
        TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    obs: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    priority: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
                    reward: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([10]),
                device=cpu,
                is_shared=False)
        >>> print(info)
        {'_weight': array([1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11,
               1.e-11, 1.e-11], dtype=float32), 'index': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}

    .. note:: Using a :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer` can smoothen the
        process of updating the priorities:

            >>> from torchrl.data.replay_buffers import TensorDictReplayBuffer as TDRB, LazyTensorStorage, PrioritizedSampler
            >>> from tensordict import TensorDict
            >>> rb = TDRB(
            ...     storage=LazyTensorStorage(10),
            ...     sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0),
            ...     priority_key="priority",  # This kwarg isn't present in regular RBs
            ... )
            >>> priority = torch.tensor([0, 1000])
            >>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
            >>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
            >>> data = torch.stack([data_0, data_1])
            >>> rb.extend(data)
            >>> rb.update_priority(data)  # Reads the "priority" key as indicated in the constructor
            >>> sample, info = rb.sample(10, return_info=True)
            >>> print(sample['index'])  # The index is packed with the tensordict
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    """

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
    ) -> None:
        if alpha <= 0:
            raise ValueError(
                f"alpha must be strictly greater than 0, got alpha={alpha}"
            )
        if beta < 0:
            raise ValueError(f"beta must be greater or equal to 0, got beta={beta}")

        self._max_capacity = max_capacity
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self.reduction = reduction
        self.dtype = dtype
        self._init()

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Samplers of type {type(self)} cannot be shared between processes."
            )
        state = copy(self.__dict__)
        return state

    def _init(self):
        if self.dtype in (torch.float, torch.FloatType, torch.float32):
            self._sum_tree = SumSegmentTreeFp32(self._max_capacity)
            self._min_tree = MinSegmentTreeFp32(self._max_capacity)
        elif self.dtype in (torch.double, torch.DoubleTensor, torch.float64):
            self._sum_tree = SumSegmentTreeFp64(self._max_capacity)
            self._min_tree = MinSegmentTreeFp64(self._max_capacity)
        else:
            raise NotImplementedError(
                f"dtype {self.dtype} not supported by PrioritizedSampler"
            )
        self._max_priority = 1.0

    def _empty(self):
        self._init()

    @property
    def default_priority(self) -> float:
        return (self._max_priority + self._eps) ** self._alpha

    def sample(self, storage: Storage, batch_size: int) -> torch.Tensor:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        p_sum = self._sum_tree.query(0, len(storage))
        p_min = self._min_tree.query(0, len(storage))
        if p_sum <= 0:
            raise RuntimeError("negative p_sum")
        if p_min <= 0:
            raise RuntimeError("negative p_min")
        # For some undefined reason, only np.random works here.
        # All PT attempts fail, even when subsequently transformed into numpy
        mass = np.random.uniform(0.0, p_sum, size=batch_size)
        # mass = torch.zeros(batch_size, dtype=torch.double).uniform_(0.0, p_sum)
        # mass = torch.rand(batch_size).mul_(p_sum)
        index = self._sum_tree.scan_lower_bound(mass)
        index = torch.as_tensor(index)
        if not index.ndim:
            index = index.unsqueeze(0)
        index.clamp_max_(len(storage) - 1)
        weight = torch.as_tensor(self._sum_tree[index])

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = torch.pow(weight / p_min, -self._beta)
        return index, {"_weight": weight}

    @torch.no_grad()
    def _add_or_extend(self, index: Union[int, torch.Tensor]) -> None:
        priority = self.default_priority

        if not (
            isinstance(priority, float)
            or len(priority) == 1
            or len(priority) == len(index)
        ):
            raise RuntimeError(
                "priority should be a scalar or an iterable of the same "
                "length as index"
            )
        # make sure everything is cast to cpu
        if isinstance(index, torch.Tensor) and not index.is_cpu:
            index = index.cpu()
        if isinstance(priority, torch.Tensor) and not priority.is_cpu:
            priority = priority.cpu()

        self._sum_tree[index] = priority
        self._min_tree[index] = priority

    def add(self, index: int) -> None:
        super().add(index)
        if index is not None:
            # some writers don't systematically write data and can return None
            self._add_or_extend(index)

    def extend(self, index: torch.Tensor) -> None:
        super().extend(index)
        if index is not None:
            # some writers don't systematically write data and can return None
            index = index.cpu()
            self._add_or_extend(index)

    @torch.no_grad()
    def update_priority(
        self, index: Union[int, torch.Tensor], priority: Union[float, torch.Tensor]
    ) -> None:
        """Updates the priority of the data pointed by the index.

        Args:
            index (int or torch.Tensor): indexes of the priorities to be
                updated.
            priority (Number or torch.Tensor): new priorities of the
                indexed elements.

        """
        priority = torch.as_tensor(priority, device=torch.device("cpu")).detach()
        index = torch.as_tensor(
            index, dtype=torch.long, device=torch.device("cpu")
        ).detach()
        # we need to reshape priority if it has more than one elements or if it has
        # a different shape than index
        if priority.numel() > 1 and priority.shape != index.shape:
            try:
                priority = priority.reshape(index.shape[:1])
            except Exception as err:
                raise RuntimeError(
                    "priority should be a number or an iterable of the same "
                    f"length as index. Got priority of shape {priority.shape} and index "
                    f"{index.shape}."
                ) from err
        elif priority.numel() <= 1:
            priority = priority.squeeze()

        self._max_priority = priority.max().clamp_min(self._max_priority).item()
        priority = torch.pow(priority + self._eps, self._alpha)
        self._sum_tree[index] = priority
        self._min_tree[index] = priority

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        self.update_priority(index, self.default_priority)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_alpha": self._alpha,
            "_beta": self._beta,
            "_eps": self._eps,
            "_max_priority": self._max_priority,
            "_sum_tree": deepcopy(self._sum_tree),
            "_min_tree": deepcopy(self._min_tree),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._alpha = state_dict["_alpha"]
        self._beta = state_dict["_beta"]
        self._eps = state_dict["_eps"]
        self._max_priority = state_dict["_max_priority"]
        self._sum_tree = state_dict.pop("_sum_tree")
        self._min_tree = state_dict.pop("_min_tree")

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        try:
            mm_st = MemoryMappedTensor.from_filename(
                shape=(self._max_capacity,),
                dtype=torch.float64,
                filename=path / "sumtree.memmap",
            )
            mm_mt = MemoryMappedTensor.from_filename(
                shape=(self._max_capacity,),
                dtype=torch.float64,
                filename=path / "mintree.memmap",
            )
        except FileNotFoundError:
            mm_st = MemoryMappedTensor.empty(
                (self._max_capacity,),
                dtype=torch.float64,
                filename=path / "sumtree.memmap",
            )
            mm_mt = MemoryMappedTensor.empty(
                (self._max_capacity,),
                dtype=torch.float64,
                filename=path / "mintree.memmap",
            )
        mm_st.copy_(
            torch.as_tensor([self._sum_tree[i] for i in range(self._max_capacity)])
        )
        mm_mt.copy_(
            torch.as_tensor([self._min_tree[i] for i in range(self._max_capacity)])
        )
        with open(path / "sampler_metadata.json", "w") as file:
            json.dump(
                {
                    "_alpha": self._alpha,
                    "_beta": self._beta,
                    "_eps": self._eps,
                    "_max_priority": self._max_priority,
                    "_max_capacity": self._max_capacity,
                },
                file,
            )

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "sampler_metadata.json", "r") as file:
            metadata = json.load(file)
        self._alpha = metadata["_alpha"]
        self._beta = metadata["_beta"]
        self._eps = metadata["_eps"]
        self._max_priority = metadata["_max_priority"]
        _max_capacity = metadata["_max_capacity"]
        if _max_capacity != self._max_capacity:
            raise RuntimeError(
                f"max capacity of loaded metadata ({_max_capacity}) differs from self._max_capacity ({self._max_capacity})."
            )
        mm_st = MemoryMappedTensor.from_filename(
            shape=(self._max_capacity,),
            dtype=torch.float64,
            filename=path / "sumtree.memmap",
        )
        mm_mt = MemoryMappedTensor.from_filename(
            shape=(self._max_capacity,),
            dtype=torch.float64,
            filename=path / "mintree.memmap",
        )
        for i, elt in enumerate(mm_st.tolist()):
            self._sum_tree[i] = elt
        for i, elt in enumerate(mm_mt.tolist()):
            self._min_tree[i] = elt


class SliceSampler(Sampler):
    """Samples slices of data along the first dimension, given start and stop signals.

    This class samples sub-trajectories with replacement. For a version without
    replacement, see :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement`.

    Keyword Args:
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        cache_values (bool, optional): to be used with static datasets.
            Will cache the start and end signal of the trajectory.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`~torchrl.collectors.split_trajectories`. Defaults to ``True``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import SliceSampler
        >>> torch.manual_seed(0)
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1_000_000),
        ...     sampler=SliceSampler(cache_values=True, num_slices=10),
        ...     batch_size=320,
        ... )
        >>> episode = torch.zeros(1000, dtype=torch.int)
        >>> episode[:300] = 1
        >>> episode[300:550] = 2
        >>> episode[550:700] = 3
        >>> episode[700:] = 4
        >>> data = TensorDict(
        ...     {
        ...         "episode": episode,
        ...         "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        ...         "act": torch.randn((20,)).expand(1000, 20),
        ...         "other": torch.randn((20, 50)).expand(1000, 20, 50),
        ...     }, [1000]
        ... )
        >>> rb.extend(data)
        >>> sample = rb.sample()
        >>> print("sample:", sample)
        >>> print("episodes", sample.get("episode").unique())
        episodes tensor([1, 2, 3, 4], dtype=torch.int32)

    :class:`~torchrl.data.replay_buffers.SliceSampler` is default-compatible with
    most of TorchRL's datasets:

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> from torchrl.data import SliceSampler
        >>>
        >>> torch.manual_seed(0)
        >>> num_slices = 10
        >>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
        >>> data = RobosetExperienceReplay(dataid, batch_size=320, sampler=SliceSampler(num_slices=num_slices))
        >>> for batch in data:
        ...     batch = batch.reshape(num_slices, -1)
        ...     break
        >>> print("check that each batch only has one episode:", batch["episode"].unique(dim=1))
        check that each batch only has one episode: tensor([[19],
                [14],
                [ 8],
                [10],
                [13],
                [ 4],
                [ 2],
                [ 3],
                [22],
                [ 8]])

    """

    def __init__(
        self,
        *,
        num_slices: int = None,
        slice_len: int = None,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
    ) -> object:
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.end_key = end_key
        self.traj_key = traj_key
        self.truncated_key = truncated_key
        self.cache_values = cache_values
        self._fetch_traj = True
        self._uses_data_prefix = False
        self.strict_length = strict_length
        self._cache = {}
        if trajectories is not None:
            if traj_key is not None or end_key:
                raise RuntimeError(
                    "`trajectories` and `end_key` or `traj_key` are exclusive arguments."
                )
            if ends is not None:
                raise RuntimeError("trajectories and ends are exclusive arguments.")
            if not cache_values:
                raise RuntimeError(
                    "To be used, trajectories requires `cache_values` to be set to `True`."
                )
            vals = self._find_start_stop_traj(trajectory=trajectories)
            self._cache["stop-and-length"] = vals

        elif ends is not None:
            if traj_key is not None or end_key:
                raise RuntimeError(
                    "`ends` and `end_key` or `traj_key` are exclusive arguments."
                )
            if trajectories is not None:
                raise RuntimeError("trajectories and ends are exclusive arguments.")
            if not cache_values:
                raise RuntimeError(
                    "To be used, ends requires `cache_values` to be set to `True`."
                )
            vals = self._find_start_stop_traj(end=ends)
            self._cache["stop-and-length"] = vals

        else:
            if end_key is None:
                end_key = ("next", "done")
            if traj_key is None:
                traj_key = "episode"
            self.end_key = end_key
            self.traj_key = traj_key

        if not ((num_slices is None) ^ (slice_len is None)):
            raise TypeError(
                "Either num_slices or slice_len must be not None, and not both. "
                f"Got num_slices={num_slices} and slice_len={slice_len}."
            )

    @staticmethod
    def _find_start_stop_traj(*, trajectory=None, end=None):
        if trajectory is not None:
            # slower
            # _, stop_idx = torch.unique_consecutive(trajectory, return_counts=True)
            # stop_idx = stop_idx.cumsum(0) - 1

            # even slower
            # t = trajectory.unsqueeze(0)
            # w = torch.tensor([1, -1], dtype=torch.int).view(1, 1, 2)
            # stop_idx = torch.conv1d(t, w).nonzero()

            # faster
            end = trajectory[:-1] != trajectory[1:]
            end = torch.cat([end, torch.ones_like(end[:1])], 0)
        else:
            end = torch.index_fill(
                end,
                index=torch.tensor(-1, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if end.ndim != 1:
            raise RuntimeError(
                f"Expected the end-of-trajectory signal to be 1-dimensional. Got a {end.ndim} tensor instead."
            )
        stop_idx = end.view(-1).nonzero().view(-1)
        start_idx = torch.cat([torch.zeros_like(stop_idx[:1]), stop_idx[:-1] + 1])
        lengths = stop_idx - start_idx + 1
        return start_idx, stop_idx, lengths

    def _tensor_slices_from_startend(self, seq_length, start):
        if isinstance(seq_length, int):
            return (
                torch.arange(
                    seq_length, device=start.device, dtype=start.dtype
                ).unsqueeze(0)
                + start.unsqueeze(1)
            ).view(-1)
        else:
            # when padding is needed
            return torch.cat(
                [
                    _start
                    + torch.arange(_seq_len, device=start.device, dtype=start.dtype)
                    for _start, _seq_len in zip(start, seq_length)
                ]
            )

    def _get_stop_and_length(self, storage, fallback=True):
        if self.cache_values and "stop-and-length" in self._cache:
            return self._cache.get("stop-and-length")

        if self._fetch_traj:
            # We first try with the traj_key
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                if isinstance(storage, TensorStorage):
                    try:
                        trajectory = storage._storage.get(self._used_traj_key)
                    except KeyError:
                        trajectory = storage._storage.get(("_data", self.traj_key))
                        # cache that value for future use
                        self._used_traj_key = ("_data", self.traj_key)
                    self._uses_data_prefix = (
                        isinstance(self._used_traj_key, tuple)
                        and self._used_traj_key[0] == "_data"
                    )
                else:
                    try:
                        trajectory = storage[:].get(self.traj_key)
                    except Exception:
                        raise RuntimeError(
                            "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
                        )
                vals = self._find_start_stop_traj(trajectory=trajectory[: len(storage)])
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = False
                    return self._get_stop_and_length(storage, fallback=False)
                raise

        else:
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                if isinstance(storage, TensorStorage):
                    try:
                        done = storage._storage.get(self._used_end_key)
                    except KeyError:
                        done = storage._storage.get(("_data", self.end_key))
                        # cache that value for future use
                        self._used_end_key = ("_data", self.end_key)
                    self._uses_data_prefix = (
                        isinstance(self._used_end_key, tuple)
                        and self._used_end_key[0] == "_data"
                    )
                else:
                    try:
                        done = storage[:].get(self.end_key)
                    except Exception:
                        raise RuntimeError(
                            "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
                        )
                vals = self._find_start_stop_traj(end=done.squeeze()[: len(storage)])
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = True
                    return self._get_stop_and_length(storage, fallback=False)
                raise

    def _adjusted_batch_size(self, batch_size):
        if self.num_slices is not None:
            if batch_size % self.num_slices != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the number of slices, got batch_size={batch_size} and num_slices={self.num_slices}."
                )
            seq_length = batch_size // self.num_slices
            num_slices = self.num_slices
        else:
            if batch_size % self.slice_len != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the slice length, got batch_size={batch_size} and slice_len={self.slice_len}."
                )
            seq_length = self.slice_len
            num_slices = batch_size // self.slice_len
        return seq_length, num_slices

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        # pick up as many trajs as we need
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        return self._sample_slices(lengths, start_idx, stop_idx, seq_length, num_slices)

    def _sample_slices(
        self, lengths, start_idx, stop_idx, seq_length, num_slices, traj_idx=None
    ) -> Tuple[torch.Tensor, dict]:
        if traj_idx is None:
            traj_idx = torch.randint(
                lengths.shape[0], (num_slices,), device=lengths.device
            )
        else:
            num_slices = traj_idx.shape[0]

        if (lengths < seq_length).any():
            if self.strict_length:
                raise RuntimeError(
                    "Some stored trajectories have a length shorter than the slice that was asked for. "
                    "Create the sampler with `strict_length=False` to allow shorter trajectories to appear "
                    "in you batch."
                )
            # make seq_length a tensor with values clamped by lengths
            seq_length = lengths[traj_idx].clamp_max(seq_length)

        relative_starts = (
            (
                torch.rand(num_slices, device=lengths.device)
                * (lengths[traj_idx] - seq_length + 1)
            )
            .floor()
            .to(start_idx.dtype)
        )
        starts = start_idx[traj_idx] + relative_starts
        index = self._tensor_slices_from_startend(seq_length, starts)
        if self.truncated_key is not None:
            truncated_key = self.truncated_key
            done_key = _replace_last(truncated_key, "done")
            terminated_key = _replace_last(truncated_key, "terminated")

            truncated = torch.zeros(
                (*index.shape, 1), dtype=torch.bool, device=index.device
            )
            if isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            traj_terminated = stop_idx[traj_idx] == start_idx[traj_idx] + seq_length - 1
            terminated = torch.zeros_like(truncated)
            if traj_terminated.any():
                if isinstance(seq_length, int):
                    truncated.view(num_slices, -1)[traj_terminated] = 1
                else:
                    truncated[(seq_length.cumsum(0) - 1)[traj_terminated]] = 1
            truncated = truncated & ~terminated
            done = terminated | truncated
            return index.to(torch.long), {
                truncated_key: truncated,
                done_key: done,
                terminated_key: terminated,
            }
        return index.to(torch.long), {}

    @property
    def _used_traj_key(self):
        return self.__dict__.get("__used_traj_key", self.traj_key)

    @_used_traj_key.setter
    def _used_traj_key(self, value):
        self.__dict__["__used_traj_key"] = value

    @property
    def _used_end_key(self):
        return self.__dict__.get("__used_end_key", self.end_key)

    @_used_end_key.setter
    def _used_end_key(self, value):
        self.__dict__["__used_end_key"] = value

    def _empty(self):
        pass

    def dumps(self, path):
        # no op - cache does not need to be saved
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_cache"] = {}
        return state


class SliceSamplerWithoutReplacement(SliceSampler, SamplerWithoutReplacement):
    """Samples slices of data along the first dimension, given start and stop signals, without replacement.

    This class is to be used with static replay buffers or in between two
    replay buffer extensions. Extending the replay buffer will reset the
    the sampler, and continuous sampling without replacement is currently not
    allowed.

    Keyword Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept.
            Defaults to ``False``.
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`~torchrl.collectors.split_trajectories`. Defaults to ``True``.
        shuffle (bool, optional): if ``False``, the order of the trajectories
            is not shuffled. Defaults to ``True``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
        >>>
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1000),
        ...     # asking for 10 slices for a total of 320 elements, ie, 10 trajectories of 32 transitions each
        ...     sampler=SliceSamplerWithoutReplacement(num_slices=10),
        ...     batch_size=320,
        ... )
        >>> episode = torch.zeros(1000, dtype=torch.int)
        >>> episode[:300] = 1
        >>> episode[300:550] = 2
        >>> episode[550:700] = 3
        >>> episode[700:] = 4
        >>> data = TensorDict(
        ...     {
        ...         "episode": episode,
        ...         "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        ...         "act": torch.randn((20,)).expand(1000, 20),
        ...         "other": torch.randn((20, 50)).expand(1000, 20, 50),
        ...     }, [1000]
        ... )
        >>> rb.extend(data)
        >>> sample = rb.sample()
        >>> # since we want trajectories of 32 transitions but there are only 4 episodes to
        >>> # sample from, we only get 4 x 32 = 128 transitions in this batch
        >>> print("sample:", sample)
        >>> print("trajectories in sample", sample.get("episode").unique())

    :class:`~torchrl.data.replay_buffers.SliceSamplerWithoutReplacement` is default-compatible with
    most of TorchRL's datasets, and allows users to consume datasets in a dataloader-like fashion:

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> from torchrl.data import SliceSamplerWithoutReplacement
        >>>
        >>> torch.manual_seed(0)
        >>> num_slices = 10
        >>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
        >>> data = RobosetExperienceReplay(dataid, batch_size=320,
        ...     sampler=SliceSamplerWithoutReplacement(num_slices=num_slices))
        >>> # the last sample is kept, since drop_last=False by default
        >>> for i, batch in enumerate(data):
        ...     print(batch.get("episode").unique())
        tensor([ 5,  6,  8, 11, 12, 14, 16, 17, 19, 24])
        tensor([ 1,  2,  7,  9, 10, 13, 15, 18, 21, 22])
        tensor([ 0,  3,  4, 20, 23])

    """

    def __init__(
        self,
        *,
        num_slices: int | None = None,
        slice_len: int | None = None,
        drop_last: bool = False,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
        shuffle: bool = True,
    ) -> object:
        SliceSampler.__init__(
            self,
            num_slices=num_slices,
            slice_len=slice_len,
            end_key=end_key,
            traj_key=traj_key,
            cache_values=True,
            truncated_key=truncated_key,
            strict_length=strict_length,
            ends=ends,
            trajectories=trajectories,
        )
        SamplerWithoutReplacement.__init__(self, drop_last=drop_last, shuffle=shuffle)

    def _empty(self):
        self._cache = {}
        SamplerWithoutReplacement._empty(self)

    def _storage_len(self, storage):
        return self._storage_len_buffer

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        self._storage_len_buffer = len(start_idx)
        # first get indices of the trajectories we want to retrieve
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        indices, _ = SamplerWithoutReplacement.sample(self, storage, num_slices)
        idx, info = self._sample_slices(
            lengths, start_idx, stop_idx, seq_length, num_slices, traj_idx=indices
        )
        return idx, info

    def state_dict(self) -> Dict[str, Any]:
        return SamplerWithoutReplacement.state_dict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return SamplerWithoutReplacement.load_state_dict(self, state_dict)


class PrioritizedSliceSampler(SliceSampler, PrioritizedSampler):
    """Samples slices of data along the first dimension, given start and stop signals, using prioritized sampling.

    This class samples sub-trajectories with replacement following a priority weighting presented in "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015.
        Prioritized experience replay."
        (https://arxiv.org/abs/1511.05952)

    For more info see :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` and :class:`~torchrl.data.replay_buffers.samplers.PrioritizedSampler`.

    Args:
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float, optional): delta added to the priorities to ensure that the buffer
            does not contain null priorities. Defaults to 1e-8.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (i.e., stored trajectory). Can be one of "max", "min",
            "median" or "mean".

    Keyword Args:
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        cache_values (bool, optional): to be used with static datasets.
            Will cache the start and end signal of the trajectory.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`~torchrl.collectors.split_trajectories`. Defaults to ``True``.

    Examples:
        >>> import torch
        >>> from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage, PrioritizedSliceSampler
        >>> from tensordict import TensorDict
        >>> sampler = PrioritizedSliceSampler(max_capacity=9, num_slices=3, alpha=0.7, beta=0.9)
        >>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(9), sampler=sampler, batch_size=6)
        >>> data = TensorDict(
        ...     {
        ...         "observation": torch.randn(9,16),
        ...         "action": torch.randn(9, 1),
        ...         "episode": torch.tensor([0,0,0,1,1,1,2,2,2], dtype=torch.long),
        ...         "steps": torch.tensor([0,1,2,0,1,2,0,1,2], dtype=torch.long),
        ...         ("next", "observation"): torch.randn(9,16),
        ...         ("next", "reward"): torch.randn(9,1),
        ...         ("next", "done"): torch.tensor([0,0,1,0,0,1,0,0,1], dtype=torch.bool).unsqueeze(1),
        ...     },
        ...     batch_size=[9],
        ... )
        >>> rb.extend(data)
        >>> sample, info = rb.sample(return_info=True)
        >>> print("episode", sample["episode"].tolist())
        episode [2, 2, 2, 2, 1, 1]
        >>> print("steps", sample["steps"].tolist())
        steps [1, 2, 0, 1, 1, 2]
        >>> print("weight", info["_weight"].tolist())
        weight [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> priority = torch.tensor([0,3,3,0,0,0,1,1,1])
        >>> rb.update_priority(torch.arange(0,9,1), priority=priority)
        >>> sample, info = rb.sample(return_info=True)
        >>> print("episode", sample["episode"].tolist())
        episode [2, 2, 2, 2, 2, 2]
        >>> print("steps", sample["steps"].tolist())
        steps [1, 2, 0, 1, 0, 1]
        >>> print("weight", info["_weight"].tolist())
        weight [9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06]
    """

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
        *,
        num_slices: int = None,
        slice_len: int = None,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
    ) -> object:
        SliceSampler.__init__(
            self,
            num_slices=num_slices,
            slice_len=slice_len,
            end_key=end_key,
            traj_key=traj_key,
            cache_values=cache_values,
            truncated_key=truncated_key,
            strict_length=strict_length,
            ends=ends,
            trajectories=trajectories,
        )
        PrioritizedSampler.__init__(
            self,
            max_capacity=max_capacity,
            alpha=alpha,
            beta=beta,
            eps=eps,
            dtype=dtype,
            reduction=reduction,
        )

    def __getstate__(self):
        state = SliceSampler.__getstate__(self)
        state.update(PrioritizedSampler.__getstate__(self))

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        # Sample `batch_size` indices representing the start of a slice.
        # The sampling is based on a weight vector.
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        seq_length, num_slices = self._adjusted_batch_size(batch_size)

        num_trajs = lengths.shape[0]
        traj_idx = torch.arange(0, num_trajs, 1, device=lengths.device)

        if (lengths < seq_length).any():
            if self.strict_length:
                raise RuntimeError(
                    "Some stored trajectories have a length shorter than the slice that was asked for. "
                    "Create the sampler with `strict_length=False` to allow shorter trajectories to appear "
                    "in you batch."
                )
            # make seq_length a tensor with values clamped by lengths
            seq_length = lengths[traj_idx].clamp_max(seq_length)

        # build a list of index that we dont want to sample: all the steps at a `seq_length` distance of
        # the end the trajectory, with the end of trajectory (`stop_idx`) included
        if isinstance(seq_length, int):
            subtractive_idx = torch.arange(
                0, seq_length - 1, 1, device=stop_idx.device, dtype=stop_idx.dtype
            )
            preceding_stop_idx = (
                stop_idx[..., None] - subtractive_idx[None, ...]
            ).view(-1)
        else:
            raise NotImplementedError("seq_length as a list is not supported for now")

        # force to not sample index at the end of a trajectory
        self._sum_tree[preceding_stop_idx] = 0.0
        # and no need to update self._min_tree

        starts, info = PrioritizedSampler.sample(
            self, storage=storage, batch_size=batch_size // seq_length
        )
        # TODO: update PrioritizedSampler.sample to return torch tensors
        starts = torch.as_tensor(starts, device=lengths.device)
        info["_weight"] = torch.as_tensor(info["_weight"], device=lengths.device)

        # extends starting indices of each slice with sequence_length to get indices of all steps
        index = self._tensor_slices_from_startend(seq_length, starts)
        # repeat the weight of each slice to match the number of steps
        info["_weight"] = torch.repeat_interleave(info["_weight"], seq_length)

        # sanity check
        if index.shape[0] != batch_size:
            raise ValueError(
                f"Number of indices is expected to match the batch size ({index.shape[0]} != {batch_size})."
            )

        if self.truncated_key is not None:
            # following logics borrowed from SliceSampler
            truncated_key = self.truncated_key
            done_key = _replace_last(truncated_key, "done")
            terminated_key = _replace_last(truncated_key, "terminated")

            truncated = torch.zeros(
                (*index.shape, 1), dtype=torch.bool, device=index.device
            )
            if isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            traj_terminated = stop_idx[traj_idx] == start_idx[traj_idx] + seq_length - 1
            terminated = torch.zeros_like(truncated)
            if traj_terminated.any():
                if isinstance(seq_length, int):
                    truncated.view(num_slices, -1)[traj_terminated] = 1
                else:
                    truncated[(seq_length.cumsum(0) - 1)[traj_terminated]] = 1
            truncated = truncated & ~terminated
            done = terminated | truncated

            info.update(
                {
                    truncated_key: truncated,
                    done_key: done,
                    terminated_key: terminated,
                }
            )
        return index.to(torch.long), info

    def _empty(self):
        # no op for SliceSampler
        PrioritizedSampler._empty(self)

    def dumps(self, path):
        # no op for SliceSampler
        PrioritizedSampler.dumps(self, path)

    def loads(self, path):
        # no op for SliceSampler
        return PrioritizedSampler.loads(self, path)

    def state_dict(self):
        # no op for SliceSampler
        return PrioritizedSampler.state_dict(self)


class SamplerEnsemble(Sampler):
    """An ensemble of samplers.

    This class is designed to work with :class:`~torchrl.data.replay_buffers.replay_buffers.ReplayBufferEnsemble`.
    It contains the samplers as well as the sampling strategy hyperparameters.

    Args:
        samplers (sequence of Sampler): the samplers to make the composite sampler.

    Keyword Args:
        p (list or tensor of probabilities, optional): if provided, indicates the
            weights of each dataset during sampling.
        sample_from_all (bool, optional): if ``True``, each dataset will be sampled
            from. This is not compatible with the ``p`` argument. Defaults to ``False``.
        num_buffer_sampled (int, optional): the number of buffers to sample.
            if ``sample_from_all=True``, this has no effect, as it defaults to the
            number of buffers. If ``sample_from_all=False``, buffers will be
            sampled according to the probabilities ``p``.

    .. warning::
      The indices provided in the info dictionary are placed in a :class:`~tensordict.TensorDict` with
      keys ``index`` and ``buffer_ids`` that allow the upper :class:`~torchrl.data.ReplayBufferEnsemble`
      and :class:`~torchrl.data.StorageEnsemble` objects to retrieve the data.
      This format is different than with other samplers which usually return indices
      as regular tensors.

    """

    def __init__(
        self, *samplers, p=None, sample_from_all=False, num_buffer_sampled=None
    ):
        self._samplers = samplers
        self.sample_from_all = sample_from_all
        if sample_from_all and p is not None:
            raise RuntimeError(
                "Cannot pass both `p` argument and `sample_from_all=True`."
            )
        self.p = p
        self.num_buffer_sampled = num_buffer_sampled

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if not isinstance(value, torch.Tensor) and value is not None:
            value = torch.tensor(value)
        if value is not None:
            value = value / value.sum().clamp_min(1e-6)
        self._p = value

    @property
    def num_buffer_sampled(self):
        value = self.__dict__.get("_num_buffer_sampled", None)
        if value is None:
            value = self.__dict__["_num_buffer_sampled"] = len(self._samplers)
        return value

    @num_buffer_sampled.setter
    def num_buffer_sampled(self, value):
        self.__dict__["_num_buffer_sampled"] = value

    def sample(self, storage, batch_size):
        if batch_size % self.num_buffer_sampled > 0:
            raise ValueError
        if not isinstance(storage, StorageEnsemble):
            raise TypeError
        sub_batch_size = batch_size // self.num_buffer_sampled
        if self.sample_from_all:
            samples, infos = zip(
                *[
                    sampler.sample(storage, sub_batch_size)
                    for storage, sampler in zip(storage._storages, self._samplers)
                ]
            )
            buffer_ids = torch.arange(len(samples))
        else:
            if self.p is None:
                buffer_ids = torch.randint(
                    len(self._samplers), (self.num_buffer_sampled,)
                )
            else:
                buffer_ids = torch.multinomial(self.p, self.num_buffer_sampled, True)
            samples, infos = zip(
                *[
                    self._samplers[i].sample(storage._storages[i], sub_batch_size)
                    for i in buffer_ids.tolist()
                ]
            )
        samples = [
            sample if isinstance(sample, torch.Tensor) else torch.tensor(sample)
            for sample in samples
        ]
        if all(samples[0].shape == sample.shape for sample in samples[1:]):
            samples_stack = torch.stack(samples)
        else:
            samples_stack = torch.nested.nested_tensor(list(samples))

        samples = TensorDict(
            {
                "index": samples_stack,
                "buffer_ids": buffer_ids,
            },
            batch_size=[self.num_buffer_sampled],
        )
        infos = torch.stack(
            [
                TensorDict.from_dict(info, batch_dims=samples.ndim - 1)
                if info
                else TensorDict({}, [])
                for info in infos
            ]
        )
        return samples, infos

    def dumps(self, path: Path):
        path = Path(path).absolute()
        for i, sampler in enumerate(self._samplers):
            sampler.dumps(path / str(i))

    def loads(self, path: Path):
        path = Path(path).absolute()
        for i, sampler in enumerate(self._samplers):
            sampler.loads(path / str(i))

    def state_dict(self) -> Dict[str, Any]:
        state_dict = OrderedDict()
        for i, sampler in enumerate(self._samplers):
            state_dict[str(i)] = sampler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for i, sampler in enumerate(self._samplers):
            sampler.load_state_dict(state_dict[str(i)])

    def _empty(self):
        raise NotImplementedError

    _INDEX_ERROR = "Expected an index of type torch.Tensor, range, np.ndarray, int, slice or ellipsis, got {} instead."

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if index[0] is Ellipsis:
                index = (slice(None), index[1:])
            result = self[index[0]]
            if len(index) > 1:
                raise IndexError(
                    f"Tuple of length greater than 1 are not accepted to index samplers of type {type(self)}."
                )
            return result
        if isinstance(index, slice) and index == slice(None):
            return self
        if isinstance(index, (list, range, np.ndarray)):
            index = torch.as_tensor(index)
        if isinstance(index, torch.Tensor):
            if index.ndim > 1:
                raise RuntimeError(
                    f"Cannot index a {type(self)} with tensor indices that have more than one dimension."
                )
            if index.is_floating_point():
                raise TypeError(
                    "A floating point index was recieved when an integer dtype was expected."
                )
        if isinstance(index, int) or (not isinstance(index, slice) and len(index) == 0):
            try:
                index = int(index)
            except Exception:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
            try:
                return self._samplers[index]
            except IndexError:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
        if isinstance(index, torch.Tensor):
            index = index.tolist()
            samplers = [self._samplers[i] for i in index]
        else:
            # slice
            samplers = self._samplers[index]
        p = self._p[index]
        return SamplerEnsemble(
            *samplers,
            p=p,
            sample_from_all=self.sample_from_all,
            num_buffer_sampled=self.num_buffer_sampled,
        )

    def __len__(self):
        return len(self._samplers)

    def __repr__(self):
        samplers = textwrap.indent(f"samplers={self._samplers}", " " * 4)
        return f"SamplerEnsemble(\n{samplers})"
