# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import textwrap
from collections.abc import Callable

from typing import Any, TYPE_CHECKING, TypeVar

import numpy as np
import torch

from tensordict import LazyStackedTensorDict, TensorDictBase
from tensordict.utils import expand_right
from torch import Tensor

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.samplers import SamplerEnsemble
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    _stack_anything,
    StorageEnsemble,
)
from torchrl.data.replay_buffers.writers import WriterEnsemble
from torchrl.envs.transforms.transforms import Transform

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


from .base import ReplayBuffer


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
            if the buffer is built explicitly.
        sample_from_all (bool, optional): if ``True``, each dataset will be sampled
            from. This is not compatible with the ``p`` argument. Defaults to ``False``.
            Can also be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
            if the buffer is built explicitly.
        num_buffer_sampled (int, optional): the number of buffers to sample.
            if ``sample_from_all=True``, this has no effect, as it defaults to the
            number of buffers. If ``sample_from_all=False``, buffers will be
            sampled according to the probabilities ``p``. Can also
            be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
            if the buffer is built explicitly.
        generator (torch.Generator, optional): a generator to use for sampling.
            Using a dedicated generator for the replay buffer can allow a fine-grained control
            over seeding, for instance keeping the global seed different but the RB seed identical
            for distributed jobs.
            Defaults to ``None`` (global default generator).

            .. warning:: As of now, the generator has no effect on the transforms.

        shared (bool, optional): whether the buffer will be shared using multiprocessing or not.
            Defaults to ``False``.
        delayed_init (bool, optional): whether to initialize storage, writer, sampler and transform
            the first time the buffer is used rather than during construction.
            This is useful when the replay buffer needs to be pickled and sent to remote workers,
            particularly when using transforms with modules that require gradients.
            If not specified, defaults to ``True`` when ``transform_factory`` is provided,
            and ``False`` otherwise.

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
        transform: Transform | None = None,  # noqa: F821
        batch_size: int | None = None,
        collate_fn: Callable | None = None,
        collate_fns: list[Callable] | None = None,
        p: Tensor = None,
        sample_from_all: bool = False,
        num_buffer_sampled: int | None = None,
        generator: torch.Generator | None = None,
        shared: bool = False,
        **kwargs,
    ):

        if collate_fn is None:
            collate_fn = _stack_anything

        if rbs:
            if storages is not None or samplers is not None or writers is not None:
                raise RuntimeError
            # Ensure all replay buffers are initialized before creating ensemble
            for rb in rbs:
                if (
                    hasattr(rb, "_delayed_init")
                    and rb._delayed_init
                    and not rb.initialized
                ):
                    rb._init()
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
            generator=generator,
            shared=shared,
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
        self, index: int | torch.Tensor | tuple | np.ndarray | list | slice | Ellipsis
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
                    "A floating point index was received when an integer dtype was expected."
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
