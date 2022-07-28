import collections
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence, Union, Tuple

import numpy as np
import torch

from ..tensordict.tensordict import TensorDictBase
from .replay_buffers import pin_memory_output, stack_tensors, stack_td
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
        self._sampler = sampler or RandomSampler()
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
    def _sample(self, batch_size: int) -> Tuple[Any, dict]:
        index, info = self._sampler.sample(self._storage, batch_size)
        data = self._storage[index]
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        return data, info

    def sample(self, batch_size: int) -> Tuple[Any, dict]:
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
    def __init__(self, priority_key: str = "td_error", **kw) -> None:
        if not kw.get("collate_fn"):

            def collate_fn(x):
                return stack_td(x, 0, contiguous=True)

            kw["collate_fn"] = collate_fn

        super().__init__(**kw)
        self.priority_key = priority_key

    def _get_priority(self, tensordict: TensorDictBase) -> Optional[torch.Tensor]:
        if self.priority_key not in tensordict.keys():
            return self._sampler.default_priority
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
        data.set("index", index)

        priority = self._get_priority(data)
        if priority:
            self.update_priority(index, priority)
        return index

    def extend(self, tensordicts: TensorDictBase) -> torch.Tensor:
        if isinstance(tensordicts, TensorDictBase):
            if tensordicts.batch_dims > 1:
                # we want the tensordict to have one dimension only. The batch size
                # of the sampled tensordicts can be changed thereafter
                if not isinstance(tensordicts, LazyStackedTensorDict):
                    tensordicts = tensordicts.clone(recursive=False)
                else:
                    tensordicts = tensordicts.contiguous()
                tensordicts.batch_size = tensordicts.batch_size[:1]
            tensordicts.set(
                "index",
                torch.zeros(
                    tensordicts.shape, device=tensordicts.device, dtype=torch.int
                ),
            )

        if not isinstance(tensordicts, TensorDictBase):
            stacked_td = torch.stack(data, 0)
        else:
            stacked_td = tensordicts

        index = super().extend(tensordicts)
        stacked_td.set(
            "index",
            torch.tensor(index, dtype=torch.int, device=stacked_td.device),
            inplace=True,
        )
        self.update_tensordict_priority(tensordicts)
        return index

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        priority = torch.tensor([self._get_priority(td) for td in data], dtype=torch.float, device=data.device)
        self.update_priority(data.get("index"), priority)
