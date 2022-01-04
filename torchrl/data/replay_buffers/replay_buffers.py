import collections
import concurrent.futures
import functools
import threading
from numbers import Number
from typing import Optional, Tuple, Union, List, Any, Callable, Iterable

from torchrl._torchrl import SumSegmentTree, MinSegmentTree

from .utils import *

__all__ = [
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "TensorDictReplayBuffer",
    "TensorDictPrioritizedReplayBuffer",
    "create_replay_buffer",
    "create_prioritized_replay_buffer"
]

from ..tensordict.tensordict import _TensorDict
from ..utils import DEVICE_TYPING


def stack_tensors(list_of_tensors: List) -> Tuple[torch.Tensor]:
    return tuple(torch.stack(tensors, 0) for tensors in zip(*list_of_tensors))


def _pin_memory(output: Any) -> Any:
    if hasattr(output, 'pin_memory'):
        return output.pin_memory()
    else:
        # print(f'object of type {type(output)} don''t have a pin_memory method')
        return output


def pin_memory_output(fun) -> Callable:
    def decorated_fun(self, *args, **kwargs):
        output = fun(self, *args, **kwargs)
        if self._pin_memory:
            _tuple_out = True
            if not isinstance(output, tuple):
                _tuple_out = False
                output = (output,)
            output = tuple(_pin_memory(_output) for _output in output)
            if _tuple_out:
                return output
            return output[0]
        return output

    return decorated_fun


class ReplayBuffer:
    def __init__(self,
                 size: int,
                 collate_fn: Optional[Callable] = None,
                 pin_memory: bool = False,
                 prefetch: Optional[int] = None):
        self._storage = []
        self._capacity = size
        self._cursor = 0
        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = stack_tensors
        self._pin_memory = pin_memory

        self._prefetch = prefetch is not None and prefetch > 0
        self._prefetch_cap = prefetch if prefetch is not None else 0
        self._prefetch_fut = collections.deque()
        if self._prefetch_cap > 0:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._prefetch_cap)

        self._replay_lock = threading.RLock()
        self._future_lock = threading.RLock()

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    @pin_memory_output
    def __getitem__(self, index: Union[int, Tensor]) -> Any:
        index = to_numpy(index)

        with self._replay_lock:
            if isinstance(index, int):
                data = self._storage[index]
            else:
                data = [self._storage[i] for i in index]

        if isinstance(data, list):
            data = self._collate_fn(data)
        return data

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def cursor(self) -> int:
        with self._replay_lock:
            return self._cursor

    def add(self, data: Any) -> int:
        with self._replay_lock:
            ret = self._cursor
            if self._cursor >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._cursor] = data
            self._cursor = (self._cursor + 1) % self._capacity
            return ret

    def extend(self, data: Iterable):
        if not len(data):
            raise Exception("extending with empty data is not supported")
        if not isinstance(data, list):
            data = list(data)
        with self._replay_lock:
            cur_size = len(self._storage)
            batch_size = len(data)
            storage = self._storage
            cursor = self._cursor
            if cur_size + batch_size <= self._capacity:
                index = np.arange(cur_size, cur_size + batch_size)
                self._storage += data
                self._cursor = (self._cursor + batch_size) % self._capacity
            elif cur_size < self._capacity:
                d = self._capacity - cur_size
                index = np.empty(batch_size, dtype=np.int64)
                index[:d] = np.arange(cur_size, self._capacity)
                index[d:] = np.arange(batch_size - d)
                storage += data[:d]
                for i, v in enumerate(data[d:]):
                    storage[i] = v
                self._cursor = batch_size - d
            elif self._cursor + batch_size <= self._capacity:
                index = np.arange(self._cursor, self._cursor + batch_size)
                for i, v in enumerate(data):
                    storage[cursor + i] = v
                self._cursor = (self._cursor + batch_size) % self._capacity
            else:
                d = self._capacity - self._cursor
                index = np.empty(batch_size, dtype=np.int64)
                index[:d] = np.arange(self._cursor, self._capacity)
                index[d:] = np.arange(batch_size - d)
                for i, v in enumerate(data[:d]):
                    storage[cursor + i] = v
                for i, v in enumerate(data[d:]):
                    storage[i] = v
                self._cursor = batch_size - d

            return index

    @pin_memory_output
    def _sample(self, batch_size: int) -> Any:
        index = np.random.randint(0, len(self._storage), size=batch_size)

        with self._replay_lock:
            data = [self._storage[i] for i in index]

        data = self._collate_fn(data)
        return data

    def sample(self, batch_size: int) -> Any:
        if not self._prefetch:
            return self._sample(batch_size)

        with self._future_lock:
            if len(self._prefetch_fut) == 0:
                ret = self._sample(batch_size)
            else:
                ret = self._prefetch_fut.popleft().result()

            while len(self._prefetch_fut) < self._prefetch_cap:
                fut = self._prefetch_executor.submit(self._sample, batch_size)
                self._prefetch_fut.append(fut)

            return ret


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 size: int,
                 alpha: float,
                 beta: float,
                 eps: float = 1e-8,
                 collate_fn=None,
                 pin_memory: bool = False,
                 prefetch: Optional[int] = None) -> None:
        super(PrioritizedReplayBuffer, self).__init__(size, collate_fn,
                                                      pin_memory, prefetch)
        assert alpha > 0
        assert beta >= 0

        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self._sum_tree = SumSegmentTree(size)
        self._min_tree = MinSegmentTree(size)
        self._max_priority = 1.0

    @pin_memory_output
    def __getitem__(self, index: Union[int,
                                       Tensor]) -> Any:
        index = to_numpy(index)

        with self._replay_lock:
            p_min = self._min_tree.query(0, self._capacity)
            assert p_min > 0
            if isinstance(index, int):
                data = self._storage[index]
                weight = np.array(self._sum_tree[index])
            else:
                data = [self._storage[i] for i in index]
                weight = self._sum_tree[index]

        if isinstance(data, list):
            data = self._collate_fn(data)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = np.power(weight / p_min, -self._beta)
        x = first_field(data)
        if isinstance(x, torch.Tensor):
            weight = to_torch(weight, x.device, self._pin_memory)
        return data, weight

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def max_priority(self) -> float:
        with self._replay_lock:
            return self._max_priority

    @property
    def _default_priority(self) -> float:
        return (self._max_priority + self._eps) ** self._alpha

    def _add_or_extend(self,
                       data: Any,
                       priority: Optional[torch.Tensor] = None,
                       do_add: bool = True) -> torch.Tensor:
        if priority is not None:
            priority = to_numpy(priority)
            max_priority = np.max(priority)
            with self._replay_lock:
                self._max_priority = max(self._max_priority, max_priority)
            priority = np.power(priority + self._eps, self._alpha)
        else:
            with self._replay_lock:
                priority = self._default_priority

        if do_add:
            index = super(PrioritizedReplayBuffer, self).add(data)
        else:
            index = super(PrioritizedReplayBuffer, self).extend(data)

        assert isinstance(
            priority,
            float) or len(priority) == 1 or len(priority) == len(index)

        with self._replay_lock:
            self._sum_tree[index] = priority
            self._min_tree[index] = priority

        return index

        # with self._replay_lock:
        #     if priority is not None:
        #         priority = data_utils.to_numpy(priority)
        #         self._max_priority = max(self._max_priority, np.max(priority))
        #         priority = np.power(priority, self._alpha)
        #     else:
        #         priority = self._max_priority ** self._alpha
        #
        #     index = super(PrioritizedReplayBuffer, self).add(data)
        #     assert isinstance(priority, float) or len(
        #         priority) == 1 or len(priority) == len(index)
        #     self._sum_tree[index] = priority
        #     self._min_tree[index] = priority
        #     return index

    def add(self, data: Any, priority: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._add_or_extend(data, priority, True)

    def extend(self, data: Iterable, priority: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._add_or_extend(data, priority, False)

    @pin_memory_output
    def _sample(self,
                batch_size: int) -> Tuple[Any, np.ndarray, torch.Tensor]:
        with self._replay_lock:
            p_sum = self._sum_tree.query(0, self._capacity)
            p_min = self._min_tree.query(0, self._capacity)
            assert p_sum > 0
            assert p_min > 0
            mass = np.random.uniform(0.0, p_sum, size=batch_size)
            index = self._sum_tree.scan_lower_bound(mass)
            if isinstance(index, torch.Tensor):
                index.clamp_max_(len(self._storage) - 1)
            else:
                index = np.clip(index, None, len(self._storage) - 1)
            data = [self._storage[i] for i in index]
            weight = self._sum_tree[index]

        data = self._collate_fn(data)

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = np.power(weight / p_min, -self._beta)

        x = first_field(data)
        if isinstance(x, torch.Tensor):
            weight = to_torch(weight, x.device, self._pin_memory)
        return data, weight, index

    def sample(self,
               batch_size: int) -> Tuple[Any, np.ndarray, torch.Tensor]:
        if not self._prefetch:
            return self._sample(batch_size)

        with self._future_lock:
            if len(self._prefetch_fut) == 0:
                ret = self._sample(batch_size)
            else:
                ret = self._prefetch_fut.popleft().result()

            while len(self._prefetch_fut) < self._prefetch_cap:
                fut = self._prefetch_executor.submit(self._sample, batch_size)
                self._prefetch_fut.append(fut)

            return ret

    def update_priority(self, index: Union[int, Tensor],
                        priority: Union[float, Tensor]) -> None:
        if isinstance(index, int):
            if not isinstance(priority, float):
                assert len(priority) == 1
                priority = priority.item()
        else:
            assert isinstance(
                priority,
                float) or len(priority) == 1 or len(index) == len(priority)
            index = to_numpy(index)
            priority = to_numpy(priority)

        with self._replay_lock:
            self._max_priority = max(self._max_priority, np.max(priority))
            priority = np.power(priority + self._eps, self._alpha)
            self._sum_tree[index] = priority
            self._min_tree[index] = priority


class TensorDictReplayBuffer(ReplayBuffer):
    def sample(self, size: int) -> Any:
        return super(TensorDictReplayBuffer, self).sample(size)[0]


class TensorDictPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self,
                 size: int,
                 alpha: float,
                 beta: float,
                 priority_key="td_error",
                 eps: float = 1e-8,
                 collate_fn=None,
                 pin_memory: bool = False,
                 prefetch: Optional[int] = None) -> None:
        if collate_fn is None:
            collate_fn = lambda x: torch.stack(x, 0)
        super(TensorDictPrioritizedReplayBuffer, self).__init__(size=size, alpha=alpha, beta=beta, eps=eps,
                                                                collate_fn=collate_fn, pin_memory=pin_memory,
                                                                prefetch=prefetch)
        self.priority_key = priority_key

    def _get_priority(self, tensor_dict: _TensorDict) -> torch.Tensor:
        assert not tensor_dict.batch_dims
        try:
            priority = tensor_dict.get(self.priority_key).item()
        except ValueError:
            raise ValueError(
                f"Found a priority key of size {tensor_dict.get(self.priority_key).shape} but expected scalar value")
        except KeyError:
            priority = self._default_priority
        return priority

    def add(self, tensor_dict: _TensorDict) -> torch.Tensor:
        priority = self._get_priority(tensor_dict)
        index = super().add(tensor_dict, priority)
        tensor_dict.set("index", index)
        return index

    def extend(self, tensor_dicts: _TensorDict) -> torch.Tensor:
        if isinstance(tensor_dicts, _TensorDict):
            try:
                priorities = tensor_dicts.get(self.priority_key)
            except KeyError:
                priorities = None
            tensor_dicts = list(tensor_dicts.unbind(0))
        else:
            priorities = [self._get_priority(td) for td in tensor_dicts]

        stacked_td = torch.stack(tensor_dicts, 0)
        idx = super().extend(tensor_dicts, priorities)
        stacked_td.set("index", idx)
        return idx

    def update_priority(self, tensor_dict: _TensorDict) -> None:
        return super().update_priority(tensor_dict.get("index"), tensor_dict.get(self.priority_key))

    def sample(self, size: int) -> _TensorDict:
        return super(TensorDictPrioritizedReplayBuffer, self).sample(size)[0]


def create_replay_buffer(size: int,
                         device: Optional[DEVICE_TYPING] = None,
                         collate_fn: Callable = None,
                         pin_memory: bool = False,
                         prefetch: Optional[int] = None) -> ReplayBuffer:
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and collate_fn is None:
        # Postman will add batch_dim for uploaded data, so using cat instead of
        # stack here.
        collate_fn = functools.partial(cat_fields_to_device,
                                       device=device)

    return ReplayBuffer(size, collate_fn, pin_memory, prefetch)


def create_prioritized_replay_buffer(size: int,
                                     alpha: Number,
                                     beta: Number,
                                     eps: float = 1e-8,
                                     device: Optional[DEVICE_TYPING] = 'cpu',
                                     collate_fn: Callable = None,
                                     pin_memory: bool = False,
                                     prefetch: Optional[int] = None) -> PrioritizedReplayBuffer:
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and collate_fn is None:
        # Postman will add batch_dim for uploaded data, so using cat instead of
        # stack here.
        collate_fn = functools.partial(cat_fields_to_device,
                                       device=device)

    return PrioritizedReplayBuffer(size, alpha, beta, eps, collate_fn,
                                   pin_memory, prefetch)
