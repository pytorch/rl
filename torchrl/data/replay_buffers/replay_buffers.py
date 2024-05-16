# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import contextlib
import json
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

import torch

from tensordict import (
    is_tensor_collection,
    is_tensorclass,
    LazyStackedTensorDict,
    NestedKey,
    TensorDictBase,
    unravel_key,
)
from tensordict.nn.utils import _set_dispatch_td_nn_modules
from tensordict.utils import expand_as_right, expand_right
from torch import Tensor

from torchrl._utils import accept_remote_rref_udf_invocation
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    Sampler,
    SamplerEnsemble,
)
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    _stack_anything,
    ListStorage,
    Storage,
    StorageEnsemble,
)
from torchrl.data.replay_buffers.utils import (
    _is_int,
    _reduce,
    _to_numpy,
    _to_torch,
    INT_CLASSES,
    pin_memory_output,
)
from torchrl.data.replay_buffers.writers import (
    RoundRobinWriter,
    TensorDictRoundRobinWriter,
    Writer,
    WriterEnsemble,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import _InvertTransform


class ReplayBuffer:
    """A generic, composable replay buffer class.

    Keyword Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided,
            a default :class:`~torchrl.data.replay_buffers.RandomSampler`
            will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.RoundRobinWriter`
            will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform, optional): Transform to be executed when
            :meth:`~.sample` is called.
            To chain transforms use the :class:`~torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. A generic callable can also be passed if the replay buffer
            is used with PyTree structures (see example below).
        batch_size (int, optional): the batch size to be used when sample() is
            called.
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.
        dim_extend (int, optional): indicates the dim to consider for
            extension when calling :meth:`~.extend`. Defaults to ``storage.ndim-1``.
            When using ``dim_extend > 0``, we recommend using the ``ndim``
            argument in the storage instantiation if that argument is
            available, to let storages know that the data is
            multi-dimensional and keep consistent notions of storage-capacity
            and batch-size during sampling.

            .. note:: This argument has no effect on :meth:`~.add` and
                therefore should be used with caution when both :meth:`~.add`
                and :meth:`~.extend` are used in a codebase. For example:

                    >>> data = torch.zeros(3, 4)
                    >>> rb = ReplayBuffer(
                    ...     storage=LazyTensorStorage(10, ndim=2),
                    ...     dim_extend=1)
                    >>> # these two approaches are equivalent:
                    >>> for d in data.unbind(1):
                    ...     rb.add(d)
                    >>> rb.extend(data)


    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import ReplayBuffer, ListStorage
        >>>
        >>> torch.manual_seed(0)
        >>> rb = ReplayBuffer(
        ...     storage=ListStorage(max_size=1000),
        ...     batch_size=5,
        ... )
        >>> # populate the replay buffer and get the item indices
        >>> data = range(10)
        >>> indices = rb.extend(data)
        >>> # sample will return as many elements as specified in the constructor
        >>> sample = rb.sample()
        >>> print(sample)
        tensor([4, 9, 3, 0, 3])
        >>> # Passing the batch-size to the sample method overrides the one in the constructor
        >>> sample = rb.sample(batch_size=3)
        >>> print(sample)
        tensor([9, 7, 3])
        >>> # one cans sample using the ``sample`` method or iterate over the buffer
        >>> for i, batch in enumerate(rb):
        ...     print(i, batch)
        ...     if i == 3:
        ...         break
        0 tensor([7, 3, 1, 6, 6])
        1 tensor([9, 8, 6, 6, 8])
        2 tensor([4, 3, 6, 9, 1])
        3 tensor([4, 4, 1, 9, 9])

    Replay buffers accept *any* kind of data. Not all storage types
    will work, as some expect numerical data only, but the default
    :class:`~torchrl.data.ListStorage` will:

    Examples:
        >>> torch.manual_seed(0)
        >>> buffer = ReplayBuffer(storage=ListStorage(100), collate_fn=lambda x: x)
        >>> indices = buffer.extend(["a", 1, None])
        >>> buffer.sample(3)
        [None, 'a', None]

    The :class:`~torchrl.data.replay_buffers.TensorStorage`, :class:`~torchrl.data.replay_buffers.LazyMemmapStorage`
    and :class:`~torchrl.data.replay_buffers.LazyTensorStorage` also work
    with any PyTree structure (a PyTree is a nested structure of arbitrary depth made of dicts,
    lists or tuples where the leaves are tensors) provided that it only contains
    tensor data.

    Examples:
        >>> from torch.utils._pytree import tree_map
        >>> def transform(x):
        ...     # Zeros all the data in the pytree
        ...     return tree_map(lambda y: y * 0, x)
        >>> rb = ReplayBuffer(storage=LazyMemmapStorage(100), transform=transform)
        >>> data = {
        ...     "a": torch.randn(3),
        ...     "b": {"c": (torch.zeros(2), [torch.ones(1)])},
        ...     30: -torch.ones(()),
        ... }
        >>> rb.add(data)
        >>> # The sample has a similar structure to the data (with a leading dimension of 10 for each tensor)
        >>> s = rb.sample(10)
        >>> # let's check that our transform did its job:
        >>> def assert0(x):
        >>>     assert (x == 0).all()
        >>> tree_map(assert0, s)

    """

    def __init__(
        self,
        *,
        storage: Storage | None = None,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "Transform" | None = None,  # noqa-F821
        batch_size: int | None = None,
        dim_extend: int | None = None,
        checkpointer: "StorageCheckpointerBase" | None = None,  # noqa: F821
    ) -> None:
        self._storage = storage if storage is not None else ListStorage(max_size=1_000)
        self._storage.attach(self)
        self._sampler = sampler if sampler is not None else RandomSampler()
        self._writer = writer if writer is not None else RoundRobinWriter()
        self._writer.register_storage(self._storage)

        self._get_collate_fn(collate_fn)
        self._pin_memory = pin_memory

        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()
        from torchrl.envs.transforms.transforms import (
            _CallableTransform,
            Compose,
            Transform,
        )

        if transform is None:
            transform = Compose()
        elif not isinstance(transform, Compose):
            if not isinstance(transform, Transform) and callable(transform):
                transform = _CallableTransform(transform)
            elif not isinstance(transform, Transform):
                raise RuntimeError(
                    "transform must be either a Transform instance or a callable."
                )
            transform = Compose(transform)
        transform.eval()
        self._transform = transform

        if batch_size is None and prefetch:
            raise ValueError(
                "Dynamic batch-size specification is incompatible "
                "with multithreaded sampling. "
                "When using prefetch, the batch-size must be specified in "
                "advance. "
            )
        if (
            batch_size is None
            and hasattr(self._sampler, "drop_last")
            and self._sampler.drop_last
        ):
            raise ValueError(
                "Samplers with drop_last=True must work with a predictible batch-size. "
                "Please pass the batch-size to the ReplayBuffer constructor."
            )
        self._batch_size = batch_size
        if dim_extend is not None and dim_extend < 0:
            raise ValueError("dim_extend must be a positive value.")
        self.dim_extend = dim_extend
        self._storage.checkpointer = checkpointer

    @property
    def dim_extend(self):
        return self._dim_extend

    @dim_extend.setter
    def dim_extend(self, value):
        if (
            hasattr(self, "_dim_extend")
            and self._dim_extend is not None
            and self._dim_extend != value
        ):
            raise RuntimeError(
                "dim_extend cannot be reset. Please create a new replay buffer."
            )

        if value is None:
            if self._storage is not None:
                ndim = self._storage.ndim
                value = ndim - 1
            else:
                value = 1

        self._dim_extend = value

    def _transpose(self, data):
        if is_tensor_collection(data):
            return data.transpose(self.dim_extend, 0)
        return torch.utils._pytree.tree_map(
            lambda x: x.transpose(self.dim_extend, 0), data
        )

    def _get_collate_fn(self, collate_fn):
        self._collate_fn = (
            collate_fn
            if collate_fn is not None
            else _get_default_collate(
                self._storage, _is_tensordict=isinstance(self, TensorDictReplayBuffer)
            )
        )

    def set_storage(self, storage: Storage, collate_fn: Callable | None = None):
        """Sets a new storage in the replay buffer and returns the previous storage.

        Args:
            storage (Storage): the new storage for the buffer.
            collate_fn (callable, optional): if provided, the collate_fn is set to this
                value. Otherwise it is reset to a default value.

        """
        prev_storage = self._storage
        self._storage = storage
        self._get_collate_fn(collate_fn)

        return prev_storage

    def set_writer(self, writer: Writer):
        """Sets a new writer in the replay buffer and returns the previous writer."""
        prev_writer = self._writer
        self._writer = writer
        self._writer.register_storage(self._storage)
        return prev_writer

    def set_sampler(self, sampler: Sampler):
        """Sets a new sampler in the replay buffer and returns the previous sampler."""
        prev_sampler = self._sampler
        self._sampler = sampler
        return prev_sampler

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    def __repr__(self) -> str:
        from torchrl.envs.transforms import Compose

        storage = textwrap.indent(f"storage={self._storage}", " " * 4)
        writer = textwrap.indent(f"writer={self._writer}", " " * 4)
        sampler = textwrap.indent(f"sampler={self._sampler}", " " * 4)
        if self._transform is not None and not (
            isinstance(self._transform, Compose) and not len(self._transform)
        ):
            transform = textwrap.indent(f"transform={self._transform}", " " * 4)
            transform = f"\n{self._transform}, "
        else:
            transform = ""
        batch_size = textwrap.indent(f"batch_size={self._batch_size}", " " * 4)
        collate_fn = textwrap.indent(f"collate_fn={self._collate_fn}", " " * 4)
        return f"{self.__class__.__name__}(\n{storage}, \n{sampler}, \n{writer}, {transform}\n{batch_size}, \n{collate_fn})"

    @pin_memory_output
    def __getitem__(self, index: int | torch.Tensor | NestedKey) -> Any:
        if isinstance(index, str) or (isinstance(index, tuple) and unravel_key(index)):
            return self[:][index]
        if isinstance(index, tuple):
            if len(index) == 1:
                return self[index[0]]
            else:
                return self[:][index]
        index = _to_numpy(index)

        if self.dim_extend > 0:
            index = (slice(None),) * self.dim_extend + (index,)
            with self._replay_lock:
                data = self._storage[index]
            data = self._transpose(data)
        else:
            with self._replay_lock:
                data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        if self._transform is not None and len(self._transform):
            with data.unlock_() if is_tensor_collection(
                data
            ) else contextlib.nullcontext():
                data = self._transform(data)

        return data

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": self._storage.state_dict(),
            "_sampler": self._sampler.state_dict(),
            "_writer": self._writer.state_dict(),
            "_transforms": self._transform.state_dict(),
            "_batch_size": self._batch_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._storage.load_state_dict(state_dict["_storage"])
        self._sampler.load_state_dict(state_dict["_sampler"])
        self._writer.load_state_dict(state_dict["_writer"])
        self._transform.load_state_dict(state_dict["_transforms"])
        self._batch_size = state_dict["_batch_size"]

    def dumps(self, path):
        """Saves the replay buffer on disk at the specified path.

        Args:
            path (Path or str): path where to save the replay buffer.

        Examples:
            >>> import tempfile
            >>> import tqdm
            >>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
            >>> from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
            >>> import torch
            >>> from tensordict import TensorDict
            >>> # Build and populate the replay buffer
            >>> S = 1_000_000
            >>> sampler = PrioritizedSampler(S, 1.1, 1.0)
            >>> # sampler = RandomSampler()
            >>> storage = LazyMemmapStorage(S)
            >>> rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)
            >>>
            >>> for _ in tqdm.tqdm(range(100)):
            ...     td = TensorDict({"obs": torch.randn(100, 3, 4), "next": {"obs": torch.randn(100, 3, 4)}, "td_error": torch.rand(100)}, [100])
            ...     rb.extend(td)
            ...     sample = rb.sample(32)
            ...     rb.update_tensordict_priority(sample)
            >>> # save and load the buffer
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     rb.dumps(tmpdir)
            ...
            ...     sampler = PrioritizedSampler(S, 1.1, 1.0)
            ...     # sampler = RandomSampler()
            ...     storage = LazyMemmapStorage(S)
            ...     rb_load = TensorDictReplayBuffer(storage=storage, sampler=sampler)
            ...     rb_load.loads(tmpdir)
            ...     assert len(rb) == len(rb_load)

        """
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        self._storage.dumps(path / "storage")
        self._sampler.dumps(path / "sampler")
        self._writer.dumps(path / "writer")
        # fall back on state_dict for transforms
        transform_sd = self._transform.state_dict()
        if transform_sd:
            torch.save(transform_sd, path / "transform.t")
        with open(path / "buffer_metadata.json", "w") as file:
            json.dump({"batch_size": self._batch_size}, file)

    def loads(self, path):
        """Loads a replay buffer state at the given path.

        The buffer should have matching components and be saved using :meth:`~.dumps`.

        Args:
            path (Path or str): path where the replay buffer was saved.

        See :meth:`~.dumps` for more info.

        """
        path = Path(path).absolute()
        self._storage.loads(path / "storage")
        self._sampler.loads(path / "sampler")
        self._writer.loads(path / "writer")
        # fall back on state_dict for transforms
        if (path / "transform.t").exists():
            self._transform.load_state_dict(torch.load(path / "transform.t"))
        with open(path / "buffer_metadata.json", "r") as file:
            metadata = json.load(file)
        self._batch_size = metadata["batch_size"]

    def save(self, *args, **kwargs):
        """Alias for :meth:`~.dumps`."""
        return self.dumps(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """Alias for :meth:`~.dumps`."""
        return self.dumps(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Alias for :meth:`~.loads`."""
        return self.loads(*args, **kwargs)

    def register_save_hook(self, hook: Callable[[Any], Any]):
        """Registers a save hook for the storage.

        .. note:: Hooks are currently not serialized when saving a replay buffer: they must
            be manually re-initialized every time the buffer is created.
        """
        self._storage.register_save_hook(hook)

    def register_load_hook(self, hook: Callable[[Any], Any]):
        """Registers a load hook for the storage.

        .. note:: Hooks are currently not serialized when saving a replay buffer: they must
            be manually re-initialized every time the buffer is created.

        """
        self._storage.register_load_hook(hook)

    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)
        return self._add(data)

    def _add(self, data):
        with self._replay_lock:
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def _extend(self, data: Sequence) -> torch.Tensor:
        with self._replay_lock:
            if self.dim_extend > 0:
                data = self._transpose(data)
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        If present, the inverse transforms will be called.`

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data added to the replay buffer.

        .. warning:: :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` can have an
          ambiguous signature when dealing with lists of values, which should be interpreted
          either as PyTree (in which case all elements in the list will be put in a slice
          in the stored PyTree in the storage) or a list of values to add one at a time.
          To solve this, TorchRL makes the clear-cut distinction between list and tuple:
          a tuple will be viewed as a PyTree, a list (at the root level) will be interpreted
          as a stack of values to add one at a time to the buffer.
          For :class:`~torchrl.data.replay_buffers.ListStorage` instances, only
          unbound elements can be provided (no PyTrees).

        """
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)
        return self._extend(data)

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
            info["index"] = index
            data = self._storage.get(index)
        # if self.dim_extend > 0:
        #     data = self._transpose(data)
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            is_td = is_tensor_collection(data)
            with data.unlock_() if is_td else contextlib.nullcontext(), _set_dispatch_td_nn_modules(
                is_td
            ):
                data = self._transform(data)

        return data, info

    def empty(self):
        """Empties the replay buffer and reset cursor to 0."""
        self._writer._empty()
        self._sampler._empty()
        self._storage._empty()

    def sample(self, batch_size: int | None = None, return_info: bool = False) -> Any:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A batch of data selected in the replay buffer.
            A tuple containing this batch and info if return_info flag is set to True.
        """
        if (
            batch_size is not None
            and self._batch_size is not None
            and batch_size != self._batch_size
        ):
            warnings.warn(
                f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
                f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments. "
                "The batch-size provided to the sample method "
                "will prevail."
            )
        elif batch_size is None and self._batch_size is not None:
            batch_size = self._batch_size
        elif batch_size is None:
            raise RuntimeError(
                "batch_size not specified. You can specify the batch_size when "
                "constructing the replay buffer, or pass it to the sample method. "
                "Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments."
            )
        if not self._prefetch:
            ret = self._sample(batch_size)
        else:
            with self._futures_lock:
                while len(self._prefetch_queue) < self._prefetch_cap:
                    fut = self._prefetch_executor.submit(self._sample, batch_size)
                    self._prefetch_queue.append(fut)
                ret = self._prefetch_queue.popleft().result()

        if return_info:
            return ret
        return ret[0]

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        self._sampler.mark_update(index)

    def append_transform(
        self, transform: "Transform", *, invert: bool = False  # noqa-F821
    ) -> ReplayBuffer:  # noqa: D417
        """Appends transform at the end.

        Transforms are applied in order when `sample` is called.

        Args:
            transform (Transform): The transform to be appended

        Keyword Args:
            invert (bool, optional): if ``True``, the transform will be inverted (forward calls will be called
                during writing and inverse calls during reading). Defaults to ``False``.

        Example:
            >>> rb = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=4)
            >>> data = TensorDict({"a": torch.zeros(10)}, [10])
            >>> def t(data):
            ...     data += 1
            ...     return data
            >>> rb.append_transform(t, invert=True)
            >>> rb.extend(data)
            >>> assert (data == 1).all()

        """
        from torchrl.envs.transforms.transforms import _CallableTransform, Transform

        if not isinstance(transform, Transform) and callable(transform):
            transform = _CallableTransform(transform)
        if invert:
            transform = _InvertTransform(transform)
        transform.eval()
        self._transform.append(transform)
        return self

    def insert_transform(
        self,
        index: int,
        transform: "Transform",  # noqa-F821
        *,
        invert: bool = False,
    ) -> ReplayBuffer:  # noqa: D417
        """Inserts transform.

        Transforms are executed in order when `sample` is called.

        Args:
            index (int): Position to insert the transform.
            transform (Transform): The transform to be appended

        Keyword Args:
            invert (bool, optional): if ``True``, the transform will be inverted (forward calls will be called
                during writing and inverse calls during reading). Defaults to ``False``.

        """
        transform.eval()
        if invert:
            transform = _InvertTransform(transform)
        self._transform.insert(index, transform)
        return self

    def __iter__(self):
        if self._sampler.ran_out:
            self._sampler.ran_out = False
        if self._batch_size is None:
            raise RuntimeError(
                "Cannot iterate over the replay buffer. "
                "Batch_size was not specified during construction of the replay buffer."
            )
        while not self._sampler.ran_out:
            yield self.sample()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        _replay_lock = state.pop("_replay_lock", None)
        _futures_lock = state.pop("_futures_lock", None)
        if _replay_lock is not None:
            state["_replay_lock_placeholder"] = None
        if _futures_lock is not None:
            state["_futures_lock_placeholder"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        if "_replay_lock_placeholder" in state:
            state.pop("_replay_lock_placeholder")
            _replay_lock = threading.RLock()
            state["_replay_lock"] = _replay_lock
        if "_futures_lock_placeholder" in state:
            state.pop("_futures_lock_placeholder")
            _futures_lock = threading.RLock()
            state["_futures_lock"] = _futures_lock
        self.__dict__.update(state)

    @property
    def sampler(self):
        """The sampler of the replay buffer.

        The sampler must be an instance of :class:`~torchrl.data.replay_buffers.Sampler`.

        """
        return self._sampler

    @property
    def writer(self):
        """The writer of the replay buffer.

        The writer must be an instance of :class:`~torchrl.data.replay_buffers.Writer`.

        """
        return self._writer

    @property
    def storage(self):
        """The storage of the replay buffer.

        The storage must be an instance of :class:`~torchrl.data.replay_buffers.Storage`.

        """
        return self._storage


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer.

    All arguments are keyword-only arguments.

    Presented in
        "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015.
        Prioritized experience replay."
        (https://arxiv.org/abs/1511.05952)

    Args:
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform, optional): Transform to be executed when
            sample() is called.
            To chain transforms use the :class:`~torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. If used with other structures, the transforms should be
            encoded with a ``"data"`` leading key that will be used to
            construct a tensordict from the non-tensordict content.
        batch_size (int, optional): the batch size to be used when sample() is
            called.
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.
        dim_extend (int, optional): indicates the dim to consider for
            extension when calling :meth:`~.extend`. Defaults to ``storage.ndim-1``.
            When using ``dim_extend > 0``, we recommend using the ``ndim``
            argument in the storage instantiation if that argument is
            available, to let storages know that the data is
            multi-dimensional and keep consistent notions of storage-capacity
            and batch-size during sampling.

            .. note:: This argument has no effect on :meth:`~.add` and
                therefore should be used with caution when both :meth:`~.add`
                and :meth:`~.extend` are used in a codebase. For example:

                    >>> data = torch.zeros(3, 4)
                    >>> rb = ReplayBuffer(
                    ...     storage=LazyTensorStorage(10, ndim=2),
                    ...     dim_extend=1)
                    >>> # these two approaches are equivalent:
                    >>> for d in data.unbind(1):
                    ...     rb.add(d)
                    >>> rb.extend(data)

    .. note::
        Generic prioritized replay buffers (ie. non-tensordict backed) require
        calling :meth:`~.sample` with the ``return_info`` argument set to
        ``True`` to have access to the indices, and hence update the priority.
        Using :class:`tensordict.TensorDict` and the related
        :class:`~torchrl.data.TensorDictPrioritizedReplayBuffer` simplifies this
        process.

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import ListStorage, PrioritizedReplayBuffer
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> rb = PrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=ListStorage(10))
        >>> data = range(10)
        >>> rb.extend(data)
        >>> sample = rb.sample(3)
        >>> print(sample)
        tensor([1, 0, 1])
        >>> # get the info to find what the indices are
        >>> sample, info = rb.sample(5, return_info=True)
        >>> print(sample, info)
        tensor([2, 7, 4, 3, 5]) {'_weight': array([1., 1., 1., 1., 1.], dtype=float32), 'index': array([2, 7, 4, 3, 5])}
        >>> # update priority
        >>> priority = torch.ones(5) * 5
        >>> rb.update_priority(info["index"], priority)
        >>> # and now a new sample, the weights should be updated
        >>> sample, info = rb.sample(5, return_info=True)
        >>> print(sample, info)
        tensor([2, 5, 2, 2, 5]) {'_weight': array([0.36278465, 0.36278465, 0.36278465, 0.36278465, 0.36278465],
              dtype=float32), 'index': array([2, 5, 2, 2, 5])}

    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        storage: Storage | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "Transform" | None = None,  # noqa-F821
        batch_size: int | None = None,
        dim_extend: int | None = None,
    ) -> None:
        if storage is None:
            storage = ListStorage(max_size=1_000)
        sampler = PrioritizedSampler(storage.max_size, alpha, beta, eps, dtype)
        super(PrioritizedReplayBuffer, self).__init__(
            storage=storage,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
            dim_extend=dim_extend,
        )


class TensorDictReplayBuffer(ReplayBuffer):
    """TensorDict-specific wrapper around the :class:`~torchrl.data.ReplayBuffer` class.

    Keyword Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.RoundRobinWriter`
            will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform, optional): Transform to be executed when
            sample() is called.
            To chain transforms use the :class:`~torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. If used with other structures, the transforms should be
            encoded with a ``"data"`` leading key that will be used to
            construct a tensordict from the non-tensordict content.
        batch_size (int, optional): the batch size to be used when sample() is
            called.
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.
        priority_key (str, optional): the key at which priority is assumed to
            be stored within TensorDicts added to this ReplayBuffer.
            This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.
            Defaults to ``"td_error"``.
        dim_extend (int, optional): indicates the dim to consider for
            extension when calling :meth:`~.extend`. Defaults to ``storage.ndim-1``.
            When using ``dim_extend > 0``, we recommend using the ``ndim``
            argument in the storage instantiation if that argument is
            available, to let storages know that the data is
            multi-dimensional and keep consistent notions of storage-capacity
            and batch-size during sampling.

            .. note:: This argument has no effect on :meth:`~.add` and
                therefore should be used with caution when both :meth:`~.add`
                and :meth:`~.extend` are used in a codebase. For example:

                    >>> data = torch.zeros(3, 4)
                    >>> rb = ReplayBuffer(
                    ...     storage=LazyTensorStorage(10, ndim=2),
                    ...     dim_extend=1)
                    >>> # these two approaches are equivalent:
                    >>> for d in data.unbind(1):
                    ...     rb.add(d)
                    >>> rb.extend(data)

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from tensordict import TensorDict
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=5)
        >>> data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 1, 1)}, [10])
        >>> rb.extend(data)
        >>> sample = rb.sample(3)
        >>> # samples keep track of the index
        >>> print(sample)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)
        >>> # we can iterate over the buffer
        >>> for i, data in enumerate(rb):
        ...     print(i, data)
        ...     if i == 2:
        ...         break
        0 TensorDict(
            fields={
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        1 TensorDict(
            fields={
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)

    """

    def __init__(self, *, priority_key: str = "td_error", **kwargs) -> None:
        writer = kwargs.get("writer", None)
        if writer is None:
            kwargs["writer"] = TensorDictRoundRobinWriter()

        super().__init__(**kwargs)
        self.priority_key = priority_key

    def _get_priority_item(self, tensordict: TensorDictBase) -> float:
        priority = tensordict.get(self.priority_key, None)
        if self._storage.ndim > 1:
            # We have to flatten the priority otherwise we'll be aggregating
            # the priority across batches
            priority = priority.flatten(0, self._storage.ndim - 1)
        if priority is None:
            return self._sampler.default_priority
        try:
            if priority.numel() > 1:
                priority = _reduce(priority, self._sampler.reduction)
            else:
                priority = priority.item()
        except ValueError:
            raise ValueError(
                f"Found a priority key of size"
                f" {tensordict.get(self.priority_key).shape} but expected "
                f"scalar value"
            )

        if self._storage.ndim > 1:
            priority = priority.unflatten(0, tensordict.shape[: self._storage.ndim])

        return priority

    def _get_priority_vector(self, tensordict: TensorDictBase) -> torch.Tensor:
        priority = tensordict.get(self.priority_key, None)
        if priority is None:
            return torch.tensor(
                self._sampler.default_priority,
                dtype=torch.float,
                device=tensordict.device,
            ).expand(tensordict.shape[0])
        if self._storage.ndim > 1:
            # We have to flatten the priority otherwise we'll be aggregating
            # the priority across batches
            priority = priority.flatten(0, self._storage.ndim - 1)

        priority = priority.reshape(priority.shape[0], -1)
        priority = _reduce(priority, self._sampler.reduction, dim=1)

        if self._storage.ndim > 1:
            priority = priority.unflatten(0, tensordict.shape[: self._storage.ndim])

        return priority

    def add(self, data: TensorDictBase) -> int:
        if self._transform is not None:
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = super()._add(data)
        if index is not None:
            if is_tensor_collection(data):
                self._set_index_in_td(data, index)

            self.update_tensordict_priority(data)
        return index

    def extend(self, tensordicts: TensorDictBase) -> torch.Tensor:
        if not isinstance(tensordicts, TensorDictBase):
            raise ValueError(
                f"{self.__class__.__name__} only accepts TensorDictBase subclasses. tensorclasses "
                f"and other types are not compatible with that class. "
                "Please use a regular `ReplayBuffer` instead."
            )
        if self._transform is not None:
            tensordicts = self._transform.inv(tensordicts)
        if tensordicts is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = super()._extend(tensordicts)
        self._set_index_in_td(tensordicts, index)
        self.update_tensordict_priority(tensordicts)
        return index

    def _set_index_in_td(self, tensordict, index):
        if index is None:
            return
        if _is_int(index):
            index = torch.as_tensor(index, device=tensordict.device)
        elif index.ndim == 2 and index.shape[:1] != tensordict.shape[:1]:
            for dim in range(2, tensordict.ndim + 1):
                if index.shape[:1].numel() == tensordict.shape[:dim].numel():
                    # if index has 2 dims and is in a non-zero format
                    index = index.unflatten(0, tensordict.shape[:dim])
                    break
            else:
                raise RuntimeError(
                    f"could not find how to reshape index with shape {index.shape} to fit in tensordict with shape {tensordict.shape}"
                )
            tensordict.set("index", index)
            return
        tensordict.set("index", expand_as_right(index, tensordict))

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        if not isinstance(self._sampler, PrioritizedSampler):
            return
        if data.ndim:
            priority = self._get_priority_vector(data)
        else:
            priority = torch.as_tensor(self._get_priority_item(data))
        index = data.get("index")
        while index.shape != priority.shape:
            # reduce index
            index = index[..., 0]
        return self.update_priority(index, priority)

    def sample(
        self,
        batch_size: int | None = None,
        return_info: bool = False,
        include_info: bool = None,
    ) -> TensorDictBase:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A tensordict containing a batch of data selected in the replay buffer.
            A tuple containing this tensordict and info if return_info flag is set to True.
        """
        if include_info is not None:
            warnings.warn(
                "include_info is going to be deprecated soon."
                "The default behaviour has changed to `include_info=True` "
                "to avoid bugs linked to wrongly preassigned values in the "
                "output tensordict."
            )

        data, info = super().sample(batch_size, return_info=True)
        is_tc = is_tensor_collection(data)
        if is_tc and not is_tensorclass(data) and include_info in (True, None):
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            for key, val in info.items():
                if key == "index" and isinstance(val, tuple):
                    val = torch.stack(val, -1)
                try:
                    val = _to_torch(val, data.device)
                    if val.ndim < data.ndim:
                        val = expand_as_right(val, data)
                    data.set(key, val)
                except RuntimeError:
                    raise RuntimeError(
                        "Failed to set the metadata (e.g., indices or weights) in the sampled tensordict within TensorDictReplayBuffer.sample. "
                        "This is probably caused by a shape mismatch (one of the transforms has proably modified "
                        "the shape of the output tensordict). "
                        "You can always recover these items from the `sample` method from a regular ReplayBuffer "
                        "instance with the 'return_info' flag set to True."
                    )
            if is_locked:
                data.lock_()
        elif not is_tc and include_info in (True, None):
            raise RuntimeError("Cannot include info in non-tensordict data")
        if return_info:
            return data, info
        return data

    @pin_memory_output
    def _sample(self, batch_size: int) -> Tuple[Any, dict]:
        with self._replay_lock:
            index, info = self._sampler.sample(self._storage, batch_size)
            info["index"] = index
            data = self._storage.get(index)
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            with data.unlock_(), _set_dispatch_td_nn_modules(True):
                data = self._transform(data)
        return data, info


class TensorDictPrioritizedReplayBuffer(TensorDictReplayBuffer):
    """TensorDict-specific wrapper around the :class:`~torchrl.data.PrioritizedReplayBuffer` class.

    This class returns tensordicts with a new key ``"index"`` that represents
    the index of each element in the replay buffer. It also provides the
    :meth:`~.update_tensordict_priority` method that only requires for the
    tensordict to be passed to it with its new priority value.

    Keyword Args:
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform, optional): Transform to be executed when
            sample() is called.
            To chain transforms use the :class:`~torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. If used with other structures, the transforms should be
            encoded with a ``"data"`` leading key that will be used to
            construct a tensordict from the non-tensordict content.
        batch_size (int, optional): the batch size to be used when sample() is
            called.
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.
        priority_key (str, optional): the key at which priority is assumed to
            be stored within TensorDicts added to this ReplayBuffer.
            This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.
            Defaults to ``"td_error"``.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectories). Can be one of "max", "min",
            "median" or "mean".
        dim_extend (int, optional): indicates the dim to consider for
            extension when calling :meth:`~.extend`. Defaults to ``storage.ndim-1``.
            When using ``dim_extend > 0``, we recommend using the ``ndim``
            argument in the storage instantiation if that argument is
            available, to let storages know that the data is
            multi-dimensional and keep consistent notions of storage-capacity
            and batch-size during sampling.

            .. note:: This argument has no effect on :meth:`~.add` and
                therefore should be used with caution when both :meth:`~.add`
                and :meth:`~.extend` are used in a codebase. For example:

                    >>> data = torch.zeros(3, 4)
                    >>> rb = ReplayBuffer(
                    ...     storage=LazyTensorStorage(10, ndim=2),
                    ...     dim_extend=1)
                    >>> # these two approaches are equivalent:
                    >>> for d in data.unbind(1):
                    ...     rb.add(d)
                    >>> rb.extend(data)

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
        >>> from tensordict import TensorDict
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> rb = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=1.1, storage=LazyTensorStorage(10), batch_size=5)
        >>> data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 3, 1)}, [10])
        >>> rb.extend(data)
        >>> print("len of rb", len(rb))
        len of rb 10
        >>> sample = rb.sample(5)
        >>> print(sample)
        TensorDict(
            fields={
                _weight: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> print("index", sample["index"])
        index tensor([9, 5, 2, 2, 7])
        >>> # give a high priority to these samples...
        >>> sample.set("td_error", 100*torch.ones(sample.shape))
        >>> # and update priority
        >>> rb.update_tensordict_priority(sample)
        >>> # the new sample should have a high overlap with the previous one
        >>> sample = rb.sample(5)
        >>> print(sample)
        TensorDict(
            fields={
                _weight: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> print("index", sample["index"])
        index tensor([2, 5, 5, 9, 7])

    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        priority_key: str = "td_error",
        eps: float = 1e-8,
        storage: Storage | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "Transform" | None = None,  # noqa-F821
        reduction: str = "max",
        batch_size: int | None = None,
        dim_extend: int | None = None,
    ) -> None:
        if storage is None:
            storage = ListStorage(max_size=1_000)
        sampler = PrioritizedSampler(
            storage.max_size, alpha, beta, eps, reduction=reduction
        )
        super(TensorDictPrioritizedReplayBuffer, self).__init__(
            priority_key=priority_key,
            storage=storage,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
            dim_extend=dim_extend,
        )


@accept_remote_rref_udf_invocation
class RemoteTensorDictReplayBuffer(TensorDictReplayBuffer):
    """A remote invocation friendly ReplayBuffer class. Public methods can be invoked by remote agents using `torch.rpc` or called locally as normal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self,
        batch_size: int | None = None,
        include_info: bool = None,
        return_info: bool = False,
    ) -> TensorDictBase:
        return super().sample(
            batch_size=batch_size, include_info=include_info, return_info=return_info
        )

    def add(self, data: TensorDictBase) -> int:
        return super().add(data)

    def extend(self, tensordicts: Union[List, TensorDictBase]) -> torch.Tensor:
        return super().extend(tensordicts)

    def update_priority(
        self, index: Union[int, torch.Tensor], priority: Union[int, torch.Tensor]
    ) -> None:
        return super().update_priority(index, priority)

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        return super().update_tensordict_priority(data)


class InPlaceSampler:
    """A sampler to write tennsordicts in-place.

    To be used cautiously as this may lead to unexpected behaviour (i.e. tensordicts
    overwritten during execution).

    """

    def __init__(self, device: DEVICE_TYPING | None = None):
        self.out = None
        if device is None:
            device = "cpu"
        self.device = torch.device(device)

    def __call__(self, list_of_tds):
        if self.out is None:
            self.out = torch.stack(list_of_tds, 0).contiguous()
            if self.device is not None:
                self.out = self.out.to(self.device)
        else:
            torch.stack(list_of_tds, 0, out=self.out)
        return self.out


def stack_tensors(list_of_tensor_iterators: List) -> Tuple[torch.Tensor]:
    """Zips a list of iterables containing tensor-like objects and stacks the resulting lists of tensors together.

    Args:
        list_of_tensor_iterators (list): Sequence containing similar iterators,
            where each element of the nested iterator is a tensor whose
            shape match the tensor of other iterators that have the same index.

    Returns:
         Tuple of stacked tensors.

    Examples:
         >>> list_of_tensor_iterators = [[torch.ones(3), torch.zeros(1,2)]
         ...     for _ in range(4)]
         >>> stack_tensors(list_of_tensor_iterators)
         (tensor([[1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.]]), tensor([[[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]]]))

    """
    return tuple(torch.stack(tensors, 0) for tensors in zip(*list_of_tensor_iterators))


class ReplayBufferEnsemble(ReplayBuffer):
    """An ensemble of replay buffers.

    This class allows to read and sample from multiple replay buffers at once.
    It automatically composes ensemble of storages (:class:`~torchrl.data.replay_buffers.storages.StorageEnsemble`),
    writers (:class:`~torchrl.data.replay_buffers.writers.WriterEnsemble`) and
    samplers (:class:`~torchrl.data.replay_buffers.samplers.SamplerEnsemble`).

    .. note::
      Writing directly to this class is forbidden, but it can be indexed to retrieve
      the nested nested-buffer and extending it.

    There are two distinct ways of constructing a :class:`~torchrl.data.ReplayBufferEnsemble`:
    one can either pass a list of replay buffers, or directly pass the components
    (storage, writers and samplers) like it is done for other replay buffer subclasses.

    Args:
        rbs (sequence of ReplayBuffer instances, optional): the replay buffers to ensemble.
        storages (StorageEnsemble, optional): the ensemble of storages, if the replay
            buffers are not passed.
        samplers (SamplerEnsemble, optional): the ensemble of samplers, if the replay
            buffers are not passed.
        writers (WriterEnsemble, optional): the ensemble of writers, if the replay
            buffers are not passed.
        transform (Transform, optional): if passed, this will be the transform
            of the ensemble of replay buffers. Individual transforms for each
            replay buffer is retrieved from its parent replay buffer, or directly
            written in the :class:`~torchrl.data.replay_buffers.storages.StorageEnsemble`
            object.
        batch_size (int, optional): the batch-size to use during sampling.
        collate_fn (callable, optional): the function to use to collate the
            data after each individual collate_fn has been called and the data
            is placed in a list (along with the buffer id).
        collate_fns (list of callables, optional): collate_fn of each nested
            replay buffer. Retrieved from the :class:`~ReplayBuffer` instances
            if not provided.
        p (list of float or Tensor, optional): a list of floating numbers
            indicating the relative weight of each replay buffer. Can also
            be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
            if the buffer is built explicitely.
        sample_from_all (bool, optional): if ``True``, each dataset will be sampled
            from. This is not compatible with the ``p`` argument. Defaults to ``False``.
            Can also be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
            if the buffer is built explicitely.
        num_buffer_sampled (int, optional): the number of buffers to sample.
            if ``sample_from_all=True``, this has no effect, as it defaults to the
            number of buffers. If ``sample_from_all=False``, buffers will be
            sampled according to the probabilities ``p``. Can also
            be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
            if the buffer is built explicitely.

    Examples:
        >>> from torchrl.envs import Compose, ToTensorImage, Resize, RenameTransform
        >>> from torchrl.data import TensorDictReplayBuffer, ReplayBufferEnsemble, LazyMemmapStorage
        >>> from tensordict import TensorDict
        >>> import torch
        >>> rb0 = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(10),
        ...     transform=Compose(
        ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
        ...         Resize(32, in_keys=["pixels", ("next", "pixels")]),
        ...         RenameTransform([("some", "key")], ["renamed"]),
        ...     ),
        ... )
        >>> rb1 = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(10),
        ...     transform=Compose(
        ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
        ...         Resize(32, in_keys=["pixels", ("next", "pixels")]),
        ...         RenameTransform(["another_key"], ["renamed"]),
        ...     ),
        ... )
        >>> rb = ReplayBufferEnsemble(
        ...     rb0,
        ...     rb1,
        ...     p=[0.5, 0.5],
        ...     transform=Resize(33, in_keys=["pixels"], out_keys=["pixels33"]),
        ... )
        >>> print(rb)
        ReplayBufferEnsemble(
            storages=StorageEnsemble(
                storages=(<torchrl.data.replay_buffers.storages.LazyMemmapStorage object at 0x13a2ef430>, <torchrl.data.replay_buffers.storages.LazyMemmapStorage object at 0x13a2f9310>),
                transforms=[Compose(
                        ToTensorImage(keys=['pixels', ('next', 'pixels')]),
                        Resize(w=32, h=32, interpolation=InterpolationMode.BILINEAR, keys=['pixels', ('next', 'pixels')]),
                        RenameTransform(keys=[('some', 'key')])), Compose(
                        ToTensorImage(keys=['pixels', ('next', 'pixels')]),
                        Resize(w=32, h=32, interpolation=InterpolationMode.BILINEAR, keys=['pixels', ('next', 'pixels')]),
                        RenameTransform(keys=['another_key']))]),
            samplers=SamplerEnsemble(
                samplers=(<torchrl.data.replay_buffers.samplers.RandomSampler object at 0x13a2f9220>, <torchrl.data.replay_buffers.samplers.RandomSampler object at 0x13a2f9f70>)),
            writers=WriterEnsemble(
                writers=(<torchrl.data.replay_buffers.writers.TensorDictRoundRobinWriter object at 0x13a2d9b50>, <torchrl.data.replay_buffers.writers.TensorDictRoundRobinWriter object at 0x13a2f95b0>)),
        batch_size=None,
        transform=Compose(
                Resize(w=33, h=33, interpolation=InterpolationMode.BILINEAR, keys=['pixels'])),
        collate_fn=<built-in method stack of type object at 0x128648260>)
        >>> data0 = TensorDict(
        ...     {
        ...         "pixels": torch.randint(255, (10, 244, 244, 3)),
        ...         ("next", "pixels"): torch.randint(255, (10, 244, 244, 3)),
        ...         ("some", "key"): torch.randn(10),
        ...     },
        ...     batch_size=[10],
        ... )
        >>> data1 = TensorDict(
        ...     {
        ...         "pixels": torch.randint(255, (10, 64, 64, 3)),
        ...         ("next", "pixels"): torch.randint(255, (10, 64, 64, 3)),
        ...         "another_key": torch.randn(10),
        ...     },
        ...     batch_size=[10],
        ... )
        >>> rb[0].extend(data0)
        >>> rb[1].extend(data1)
        >>> for _ in range(2):
        ...     sample = rb.sample(10)
        ...     assert sample["next", "pixels"].shape == torch.Size([2, 5, 3, 32, 32])
        ...     assert sample["pixels"].shape == torch.Size([2, 5, 3, 32, 32])
        ...     assert sample["pixels33"].shape == torch.Size([2, 5, 3, 33, 33])
        ...     assert sample["renamed"].shape == torch.Size([2, 5])

    """

    _collate_fn_val = None

    def __init__(
        self,
        *rbs,
        storages: StorageEnsemble | None = None,
        samplers: SamplerEnsemble | None = None,
        writers: WriterEnsemble | None = None,
        transform: "Transform" | None = None,  # noqa: F821
        batch_size: int | None = None,
        collate_fn: Callable | None = None,
        collate_fns: List[Callable] | None = None,
        p: Tensor = None,
        sample_from_all: bool = False,
        num_buffer_sampled: int | None = None,
        **kwargs,
    ):

        if collate_fn is None:
            collate_fn = _stack_anything

        if rbs:
            if storages is not None or samplers is not None or writers is not None:
                raise RuntimeError
            storages = StorageEnsemble(
                *[rb._storage for rb in rbs], transforms=[rb._transform for rb in rbs]
            )
            samplers = SamplerEnsemble(
                *[rb._sampler for rb in rbs],
                p=p,
                sample_from_all=sample_from_all,
                num_buffer_sampled=num_buffer_sampled,
            )
            writers = WriterEnsemble(*[rb._writer for rb in rbs])
            if collate_fns is None:
                collate_fns = [rb._collate_fn for rb in rbs]
        else:
            rbs = None
            if collate_fns is None:
                collate_fns = [
                    _get_default_collate(storage) for storage in storages._storages
                ]
        self._rbs = rbs
        self._collate_fns = collate_fns
        super().__init__(
            storage=storages,
            sampler=samplers,
            writer=writers,
            transform=transform,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **kwargs,
        )

    def _sample(self, *args, **kwargs):
        sample, info = super()._sample(*args, **kwargs)
        if isinstance(sample, TensorDictBase):
            buffer_ids = info.get(("index", "buffer_ids"))
            info.set(
                ("index", "buffer_ids"), expand_right(buffer_ids, sample.batch_size)
            )
            if isinstance(info, LazyStackedTensorDict):
                for _info, _sample in zip(
                    info.unbind(info.stack_dim), sample.unbind(info.stack_dim)
                ):
                    _info.batch_size = _sample.batch_size
                info = torch.stack(info.tensordicts, info.stack_dim)
            else:
                info.batch_size = sample.batch_size
            sample.update(info)

        return sample, info

    @property
    def _collate_fn(self):
        def new_collate(samples):
            samples = [self._collate_fns[i](sample) for (i, sample) in samples]
            return self._collate_fn_val(samples)

        return new_collate

    @_collate_fn.setter
    def _collate_fn(self, value):
        self._collate_fn_val = value

    _INDEX_ERROR = "Expected an index of type torch.Tensor, range, np.ndarray, int, slice or ellipsis, got {} instead."

    def __getitem__(
        self, index: Union[int, torch.Tensor, Tuple, np.ndarray, List, slice, Ellipsis]
    ) -> Any:
        # accepts inputs:
        # (int | 1d tensor | 1d list | 1d array | slice | ellipsis | range, int | tensor | list | array | slice | ellipsis | range)
        # tensor
        if isinstance(index, tuple):
            if index[0] is Ellipsis:
                index = (slice(None), index[1:])
            rb = self[index[0]]
            if len(index) > 1:
                if rb is self:
                    # then index[0] is an ellipsis/slice(None)
                    sample = [
                        (i, storage[index[1:]])
                        for i, storage in enumerate(self._storage._storages)
                    ]
                    return self._collate_fn(sample)
                if isinstance(rb, ReplayBufferEnsemble):
                    new_index = (slice(None), *index[1:])
                    return rb[new_index]
                return rb[index[1:]]
            return rb
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
        if self._rbs is not None and (
            isinstance(index, int) or (not isinstance(index, slice) and len(index) == 0)
        ):
            try:
                index = int(index)
            except Exception:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
            try:
                return self._rbs[index]
            except IndexError:
                raise IndexError(self._INDEX_ERROR.format(type(index)))

        if self._rbs is not None:
            if isinstance(index, torch.Tensor):
                index = index.tolist()
                rbs = [self._rbs[i] for i in index]
                _collate_fns = [self._collate_fns[i] for i in index]
            else:
                try:
                    # slice
                    rbs = self._rbs[index]
                    _collate_fns = self._collate_fns[index]
                except IndexError:
                    raise IndexError(self._INDEX_ERROR.format(type(index)))
            p = self._sampler._p[index] if self._sampler._p is not None else None
            return ReplayBufferEnsemble(
                *rbs,
                transform=self._transform,
                batch_size=self._batch_size,
                collate_fn=self._collate_fn_val,
                collate_fns=_collate_fns,
                sample_from_all=self._sampler.sample_from_all,
                num_buffer_sampled=self._sampler.num_buffer_sampled,
                p=p,
            )

        try:
            samplers = self._sampler[index]
            writers = self._writer[index]
            storages = self._storage[index]
            if isinstance(index, torch.Tensor):
                _collate_fns = [self._collate_fns[i] for i in index.tolist()]
            else:
                _collate_fns = self._collate_fns[index]
            p = self._sampler._p[index] if self._sampler._p is not None else None

        except IndexError:
            raise IndexError(self._INDEX_ERROR.format(type(index)))

        return ReplayBufferEnsemble(
            samplers=samplers,
            writers=writers,
            storages=storages,
            transform=self._transform,
            batch_size=self._batch_size,
            collate_fn=self._collate_fn_val,
            collate_fns=_collate_fns,
            sample_from_all=self._sampler.sample_from_all,
            num_buffer_sampled=self._sampler.num_buffer_sampled,
            p=p,
        )

    def __len__(self):
        return len(self._storage)

    def __repr__(self):
        storages = textwrap.indent(f"storages={self._storage}", " " * 4)
        writers = textwrap.indent(f"writers={self._writer}", " " * 4)
        samplers = textwrap.indent(f"samplers={self._sampler}", " " * 4)
        return f"ReplayBufferEnsemble(\n{storages}, \n{samplers}, \n{writers}, \nbatch_size={self._batch_size}, \ntransform={self._transform}, \ncollate_fn={self._collate_fn_val})"
