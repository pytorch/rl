import collections
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence, Union, Tuple, List

import torch

from ..tensordict.tensordict import TensorDictBase, LazyStackedTensorDict
from .replay_buffers import pin_memory_output, stack_tensors, stack_td
from .samplers import Sampler, RandomSampler
from .storages import Storage, ListStorage
from .utils import INT_CLASSES, to_numpy
from .writers import Writer, RoundRobinWriter


class ReplayBuffer:
    """
    #TODO: Description of the ReplayBuffer class needed.
    Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default ListStorage with max_size of 1_000 will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default RoundRobinWriter() will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
    """

    def __init__(
        self,
        storage: Optional[Storage] = None,
        sampler: Optional[Sampler] = None,
        writer: Optional[Writer] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
    ) -> None:
        self._storage = storage if storage is not None else ListStorage(max_size=1_000)
        self._sampler = sampler if sampler is not None else RandomSampler()
        self._writer = writer if writer is not None else RoundRobinWriter()
        self._writer.register_storage(self._storage)

        self._collate_fn = collate_fn or stack_tensors
        self._pin_memory = pin_memory

        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"storage={self._storage}, "
            f"sampler={self._sampler}, "
            f"writer={self._writer}"
            ")"
        )

    @pin_memory_output
    def __getitem__(self, index: Union[int, torch.Tensor]) -> Any:
        index = to_numpy(index)
        with self._replay_lock:
            data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        return data

    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        with self._replay_lock:
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in
        an iterable.

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data aded to the replay buffer.
        """
        with self._replay_lock:
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    def update_priority(
        self,
        index: Union[int, torch.Tensor],
        priority: Union[int, torch.Tensor],
    ) -> None:
        with self._replay_lock:
            self._sampler.update_priority(index, priority)

    @pin_memory_output
    def _sample(self, batch_size: int) -> Tuple[Any, dict]:
        with self._replay_lock:
            index, info = self._sampler.sample(self._storage, batch_size)
            data = self._storage[index]
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        return data, info

    def sample(self, batch_size: int) -> Tuple[Any, dict]:
        """
        Samples a batch of data from the replay buffer.
        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int): size of data to be collected.

        Returns:
            A batch of data selected in the replay buffer.
        """
        if not self._prefetch:
            return self._sample(batch_size)

        if len(self._prefetch_queue) == 0:
            ret = self._sample(batch_size)
        else:
            with self._futures_lock:
                ret = self._prefetch_queue.popleft().result()

        with self._futures_lock:
            while len(self._prefetch_queue) < self._prefetch_cap:
                fut = self._prefetch_executor.submit(self._sample, batch_size)
                self._prefetch_queue.append(fut)

        return ret


class TensorDictReplayBuffer(ReplayBuffer):
    """
    TensorDict-specific wrapper around the ReplayBuffer class.
    Args:
        priority_key (str): the key at which priority is assumed to be stored
            within TensorDicts added to this ReplayBuffer.
    """

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
            tensordict = tensordict.clone(recurse=False)
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

    def extend(self, tensordicts: Union[List, TensorDictBase]) -> torch.Tensor:
        if isinstance(tensordicts, TensorDictBase):
            if tensordicts.batch_dims > 1:
                # we want the tensordict to have one dimension only. The batch size
                # of the sampled tensordicts can be changed thereafter
                if not isinstance(tensordicts, LazyStackedTensorDict):
                    tensordicts = tensordicts.clone(recurse=False)
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
            stacked_td = torch.stack(tensordicts, 0)
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
        priority = torch.tensor(
            [self._get_priority(td) for td in data],
            dtype=torch.float,
            device=data.device,
        )
        self.update_priority(data.get("index"), priority)

    def sample(self, batch_size: int, include_info: bool = False) -> TensorDictBase:
        data, info = super().sample(batch_size)
        if include_info:
            for k, v in info.items():
                data.set(k, torch.tensor(v, device=data.device), inplace=True)
        return data, info
