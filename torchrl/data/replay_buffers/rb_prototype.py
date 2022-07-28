import collections
from typing import Any, Callable, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor

import torch

from ..tensordict.tensordict import TensorDictBase
from .replay_buffers import pin_memory_output, stack_tensors
from .samplers import Sampler, RandomSampler
from .storages import Storage, ListStorage
from .utils import INT_CLASSES, to_numpy
from .writers import Writer, RoundRobinWriter


class ReplayBuffer:
    def __init__(
        self,
        storage: Storage = None,
        sampler: Sampler = None,
        writer: Writer = None,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        prefetch: int = None,
    ) -> None:
        self._storage = storage or ListStorage(size=1_000)
        self._sampler = sampler or RandomSampler(max_capacity=self._storage.size)
        self._writer = writer or RoundRobinWriter()
        self._writer.register_storage(self._storage)

        self._collate_fn = collate_fn or stack_tensors
        self._pin_memory = pin_memory

        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(storage={self._storage}, sampler={self._sampler}, writer={self._writer})"

    @pin_memory_output
    def __getitem__(self, index: Union[int, torch.Tensor]) -> Any:
        index = to_numpy(index)
        data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        return data

    def add(self, data: Any) -> int:
        index = self._writer.add(data)
        self._sampler.add(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        index = self._writer.extend(data)
        self._sampler.extend(index)
        return index

    def update_priority(
        self,
        index: Union[int, torch.Tensor],
        priority: Union[int, torch.Tensor],
    ) -> None:
        self._sampler.update_priority(index, priority)

    @pin_memory_output
    def _sample(self, batch_size: int) -> Any:
        index = self._sampler.sample(batch_size)
        data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        return data

    def sample(self, batch_size: int) -> Any:
        if not self._prefetch:
            return self._sample(batch_size)

        if len(self._prefetch_fut) == 0:
            ret = self._sample(batch_size)
        else:
            ret = self._prefetch_queue.popleft().result()

        while len(self._prefetch_queue) < self._prefetch_cap:
            fut = self._prefetch_executor.submit(self._sample, batch_size)
            self._prefetch_queue.append(fut)

        return ret


class TensorDictReplayBuffer(ReplayBuffer):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)

        if not self._collate_fn:
            def collate_fn(x):
                return stack_td(x, 0, contiguous=True)

            self._collate_fn = collate_fn

    def _get_priority(self, tensordict: TensorDictBase) -> Optional[torch.Tensor]:
        if self.priority_key not in tensordict.keys():
            return None
        if tensordict.batch_dims:
            tensordict = tensordict.clone(recursive=False)
            tensordict.batch_size = []
        try:
            priority = tensordict.get(self.priority_key).item()
        except ValueError:
            raise ValueError(
                f"Found a priority key of size"
                f" {tensordict.get(self.priority_key).shape} but expected "
                f"scalar value"
            )
        return priority

    def add(self, data: TensorDictBase) -> int:
        index = super().add(data)
        data.set("index", index, inplace=True)

        priority = self._get_priority(data)
        if priority:
            self.update_priority(index, priority)

    def extend(self, data: torch.Tensor) -> torch.Tensor:
        index = super().extend(data)
        data.set(
            "index",
            torch.tensor(index, dtype=torch.int, device=data.device),
            inplace=True,
        )
        priorities = [self._get_priority(td) or self._default_priority for td in data]
        self.update_priority(index, priorities)
