from abc import ABC, abstractmethod
from typing import Any, Union, Tuple

import numpy as np
import torch

from torchrl._torchrl import (
    MinSegmentTreeFp32,
    MinSegmentTreeFp64,
    SumSegmentTreeFp32,
    SumSegmentTreeFp64,
)
from .storages import Storage
from .utils import INT_CLASSES, _to_numpy


class Sampler(ABC):
    """A generic sampler base class for composable Replay Buffers."""

    @abstractmethod
    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        raise NotImplementedError

    def add(self, index: int) -> None:
        pass

    def extend(self, index: torch.Tensor) -> None:
        pass

    def update_priority(
        self, index: Union[int, torch.Tensor], priority: Union[int, torch.Tensor]
    ) -> dict:
        pass

    @property
    def default_priority(self) -> float:
        return 1.0


class RandomSampler(Sampler):
    """A uniformly random sampler for composable replay buffers."""

    def sample(self, storage: Storage, batch_size: int) -> Tuple[np.array, dict]:
        index = np.random.randint(0, len(storage), size=batch_size)
        return index, {}


class PrioritizedSampler(Sampler):
    """Prioritized sampler for replay buffer.

    Presented in "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015.
        Prioritized experience replay."
        (https://arxiv.org/abs/1511.05952)

    Args:
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.

    """

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
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
        if dtype in (torch.float, torch.FloatType, torch.float32):
            self._sum_tree = SumSegmentTreeFp32(self._max_capacity)
            self._min_tree = MinSegmentTreeFp32(self._max_capacity)
        elif dtype in (torch.double, torch.DoubleTensor, torch.float64):
            self._sum_tree = SumSegmentTreeFp64(self._max_capacity)
            self._min_tree = MinSegmentTreeFp64(self._max_capacity)
        else:
            raise NotImplementedError(
                f"dtype {dtype} not supported by PrioritizedReplayBuffer"
            )
        self._max_priority = 1.0

    @property
    def default_priority(self) -> float:
        return (self._max_priority + self._eps) ** self._alpha

    def sample(self, storage: Storage, batch_size: int) -> torch.Tensor:
        p_sum = self._sum_tree.query(0, len(storage))
        p_min = self._min_tree.query(0, len(storage))
        if p_sum <= 0:
            raise RuntimeError("negative p_sum")
        if p_min <= 0:
            raise RuntimeError("negative p_min")
        mass = np.random.uniform(0.0, p_sum, size=batch_size)
        index = self._sum_tree.scan_lower_bound(mass)
        if isinstance(index, torch.Tensor):
            index.clamp_max_(len(storage) - 1)
        else:
            index = np.clip(index, None, len(storage) - 1)
        weight = self._sum_tree[index]

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = np.power(weight / p_min, -self._beta)
        return index, {"_weight": weight}

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

        self._sum_tree[index] = priority
        self._min_tree[index] = priority

    def add(self, index: int) -> None:
        super().add(index)
        self._add_or_extend(index)

    def extend(self, index: torch.Tensor) -> None:
        super().extend(index)
        self._add_or_extend(index)

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
        if isinstance(index, INT_CLASSES):
            if not isinstance(priority, float):
                if len(priority) != 1:
                    raise RuntimeError(
                        f"priority length should be 1, got {len(priority)}"
                    )
                priority = priority.item()
        else:
            if not (
                isinstance(priority, float)
                or len(priority) == 1
                or len(index) == len(priority)
            ):
                raise RuntimeError(
                    "priority should be a number or an iterable of the same "
                    "length as index"
                )
            index = _to_numpy(index)
            priority = _to_numpy(priority)

        self._max_priority = max(self._max_priority, np.max(priority))
        priority = np.power(priority + self._eps, self._alpha)
        self._sum_tree[index] = priority
        self._min_tree[index] = priority
