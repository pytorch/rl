# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping
from contextlib import ExitStack

from typing import Any, Literal, TYPE_CHECKING, TypeVar

import numpy as np
import torch

from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NestedKey,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.nn.utils import _set_dispatch_td_nn_modules
from tensordict.utils import expand_right
from torch import Tensor

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.samplers import SamplerEnsemble, SliceSampler
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    _stack_anything,
    StorageEnsemble,
)
from torchrl.data.replay_buffers.writers import RoundRobinWriter, WriterEnsemble
from torchrl.envs.transforms.transforms import Transform

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


from .base import ConditionalUpdateResult, ReplayBuffer


class ReplayBufferEnsemble(ReplayBuffer):
    """An ensemble of replay buffers.

    This class allows to read and sample from multiple replay buffers at once.
    It automatically composes ensemble of storages (:class:`~torchrl.data.replay_buffers.storages.StorageEnsemble`),
    writers (:class:`~torchrl.data.replay_buffers.writers.WriterEnsemble`) and
    samplers (:class:`~torchrl.data.replay_buffers.samplers.SamplerEnsemble`).

    .. note::
      Writing directly to this class is disabled by default. Pass exactly one
      of ``routing_key`` or ``routing_dim`` to enable routed writes while
      preserving the historical read-only behavior for existing ensembles.

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
        p (list of float, Tensor, or ``"sampleable"``, optional): relative
            weights of each replay buffer. ``"sampleable"`` dynamically
            weights members by their available records or slice windows and
            excludes members that are not ready.
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
        routing_key (NestedKey, optional): key containing a member id for each
            record. Routed input is flattened, grouped stably by member and
            written to the corresponding nested replay buffer.
        routing_dim (int, optional): batch dimension whose entries correspond
            to ensemble members. The dimension size must equal the number of
            members. Exclusive with ``routing_key``.
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
        p: Tensor | list[float] | Literal["sampleable"] | None = None,
        sample_from_all: bool = False,
        num_buffer_sampled: int | None = None,
        routing_key: NestedKey | None = None,
        routing_dim: int | None = None,
        generator: torch.Generator | None = None,
        shared: bool = False,
        **kwargs,
    ):

        if routing_key is not None and routing_dim is not None:
            raise ValueError("routing_key and routing_dim are mutually exclusive.")
        if routing_key is not None:
            routing_key = unravel_key(routing_key)
        if routing_dim is not None and not isinstance(routing_dim, int):
            raise TypeError("routing_dim must be an integer.")

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
            if collate_fns is None:
                collate_fns = [
                    _get_default_collate(storage) for storage in storages._storages
                ]
            transforms = storages._transforms or [None] * len(storages._storages)
            rbs = tuple(
                ReplayBuffer(
                    storage=storage,
                    sampler=sampler,
                    writer=writer,
                    transform=member_transform,
                    collate_fn=member_collate,
                    checkpointer=storage.checkpointer,
                )
                for storage, sampler, writer, member_transform, member_collate in zip(
                    storages._storages,
                    samplers._samplers,
                    writers._writers,
                    transforms,
                    collate_fns,
                )
            )
        self._rbs = rbs
        self._collate_fns = collate_fns
        self.routing_key = routing_key
        self.routing_dim = routing_dim
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

    def _route(self, data: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor]:
        if not isinstance(data, TensorDictBase):
            raise TypeError("Routed replay writes require a TensorDict input.")
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        flat_data = data.reshape(-1)
        if self.routing_key is not None:
            buffer_ids = flat_data.get(self.routing_key)
            while buffer_ids.ndim > 1 and buffer_ids.shape[-1] == 1:
                buffer_ids = buffer_ids.squeeze(-1)
            if buffer_ids.shape != flat_data.batch_size:
                raise ValueError(
                    f"routing_key {self.routing_key!r} must contain one member id "
                    f"per record, got shape {tuple(buffer_ids.shape)} for "
                    f"batch size {tuple(flat_data.batch_size)}."
                )
        elif self.routing_dim is not None:
            if not data.ndim:
                raise ValueError("routing_dim requires batched input.")
            routing_dim = self.routing_dim % data.ndim
            if data.shape[routing_dim] != len(self._storage._storages):
                raise ValueError(
                    f"routing_dim {self.routing_dim} has size "
                    f"{data.shape[routing_dim]}, expected "
                    f"{len(self._storage._storages)} ensemble members."
                )
            id_shape = [1] * data.ndim
            id_shape[routing_dim] = data.shape[routing_dim]
            buffer_ids = (
                torch.arange(data.shape[routing_dim], device=data.device)
                .reshape(id_shape)
                .expand(data.batch_size)
                .reshape(-1)
            )
        else:
            raise RuntimeError(
                "ReplayBufferEnsemble writes are disabled. Configure routing_key "
                "or routing_dim to enable routed writes."
            )
        if buffer_ids.dtype == torch.bool or buffer_ids.is_floating_point():
            raise TypeError("ReplayBufferEnsemble member ids must be integers.")
        buffer_ids = buffer_ids.to(torch.long)
        if buffer_ids.numel() and (
            (buffer_ids < 0).any() or (buffer_ids >= len(self._storage._storages)).any()
        ):
            raise ValueError(
                f"ReplayBufferEnsemble member ids must lie in [0, "
                f"{len(self._storage._storages) - 1}]."
            )
        return flat_data, buffer_ids

    def extend(
        self, data: TensorDictBase, *, update_priority: bool | None = None
    ) -> TensorDictBase:
        """Routes and writes a batch, returning member-local write metadata.

        The returned tensordict is flat and aligned with ``data.reshape(-1)``.
        It contains ``"buffer_ids"`` and member-local ``"index"`` entries,
        plus ``"index_generation"`` when every member writer tracks
        generations.
        """
        if update_priority is not None:
            raise NotImplementedError(
                "update_priority is not supported by routed ensemble writes."
            )
        flat_data, buffer_ids = self._route(data)
        if not buffer_ids.numel():
            return TensorDict(
                {
                    "buffer_ids": buffer_ids,
                    "index": torch.empty(0, dtype=torch.long, device=buffer_ids.device),
                },
                batch_size=[0],
            )

        local_indices = None
        local_generations = None
        with self._replay_lock, self._write_lock:
            for member_id, member_buffer in enumerate(self._rbs):
                positions = (buffer_ids == member_id).nonzero().flatten()
                if not positions.numel():
                    continue
                member_index = member_buffer.extend(flat_data[positions])
                if isinstance(member_index, tuple):
                    member_index = torch.stack(member_index, -1)
                else:
                    member_index = torch.as_tensor(member_index)
                if member_index.ndim == 0:
                    member_index = member_index.unsqueeze(0)
                member_index = member_index.to(buffer_ids.device)
                if local_indices is None:
                    local_indices = torch.empty(
                        (buffer_ids.numel(), *member_index.shape[1:]),
                        dtype=member_index.dtype,
                        device=buffer_ids.device,
                    )
                elif member_index.shape[1:] != local_indices.shape[1:]:
                    raise RuntimeError(
                        "All routed replay-buffer members must use compatible "
                        "index shapes."
                    )
                local_indices[positions] = member_index

                if self._writer.tracks_generations:
                    generation = member_buffer.writer.generations_of(member_index)
                    slots = (
                        member_index[..., 0]
                        if member_buffer.storage.ndim > 1
                        else member_index
                    ).reshape(-1)
                    if slots.unique().numel() != slots.numel():
                        later_occurrences = torch.empty_like(slots)
                        seen = {}
                        for position in range(slots.numel() - 1, -1, -1):
                            slot = int(slots[position].cpu())
                            later_occurrences[position] = seen.get(slot, 0)
                            seen[slot] = seen.get(slot, 0) + 1
                        generation = generation.reshape(-1) - later_occurrences.to(
                            generation.device
                        )
                    generation = generation.to(buffer_ids.device)
                    if local_generations is None:
                        local_generations = torch.empty(
                            buffer_ids.numel(),
                            dtype=generation.dtype,
                            device=buffer_ids.device,
                        )
                    local_generations[positions] = generation.reshape(-1)

        metadata = TensorDict(
            {"buffer_ids": buffer_ids, "index": local_indices},
            batch_size=[buffer_ids.numel()],
        )
        if local_generations is not None:
            metadata.set("index_generation", local_generations)
        return metadata

    def add(self, data: TensorDictBase) -> TensorDictBase:
        """Routes one record through ``routing_key`` and returns its metadata."""
        if self.routing_dim is not None:
            raise RuntimeError("Use extend() for routing_dim writes.")
        if data.ndim:
            raise ValueError("add() expects a scalar TensorDict record.")
        return self.extend(data.unsqueeze(0))[0]

    def _conditional_update_device(self) -> torch.device | None:
        devices = {member._conditional_update_device() for member in self._rbs}
        return devices.pop() if len(devices) == 1 else None

    def update_if_present(
        self,
        *,
        index: TensorDictBase,
        generation: torch.Tensor,
        patch: Mapping[NestedKey, torch.Tensor] | TensorDictBase,
        version_key: NestedKey | None = None,
        version: int | torch.Tensor | None = None,
        require_newer: bool = False,
    ) -> ConditionalUpdateResult:
        """Routes a conditional update to the member named by each handle.

        The patch moves to the members' common storage device once, before the
        records are grouped by member, so the per-member updates never copy
        across devices.
        """
        if not isinstance(index, TensorDictBase):
            raise TypeError(
                "ReplayBufferEnsemble conditional updates require routed index metadata."
            )
        buffer_ids = index.get("buffer_ids")
        local_index = index.get("index")
        leading_shape = buffer_ids.shape
        if local_index.shape[: len(leading_shape)] != leading_shape:
            raise ValueError(
                "Member-local indices must start with the buffer-id shape, got "
                f"{tuple(local_index.shape)} and {tuple(leading_shape)}."
            )
        if buffer_ids.dtype == torch.bool or buffer_ids.is_floating_point():
            raise TypeError("ReplayBufferEnsemble member ids must be integers.")
        num_records = buffer_ids.numel()
        flat_buffer_ids = buffer_ids.reshape(-1)
        if num_records and (
            (flat_buffer_ids < 0).any() or (flat_buffer_ids >= len(self._rbs)).any()
        ):
            raise ValueError(
                f"ReplayBufferEnsemble member ids must lie in [0, "
                f"{len(self._rbs) - 1}]."
            )
        if (version_key is None) != (version is None):
            raise ValueError("version_key and version must be provided together.")
        if not self._writer.tracks_generations:
            raise RuntimeError(
                "Conditional updates require every ensemble member writer to "
                "track slot generations."
            )
        if any(
            not getattr(member.storage, "supports_conditional_update", False)
            for member in self._rbs
        ):
            raise RuntimeError(
                "Conditional updates require every ensemble member storage to "
                "support conditional patches."
            )
        flat_local_index = local_index.reshape(
            num_records, *local_index.shape[len(leading_shape) :]
        )
        flat_generation = generation.reshape(-1)
        if flat_generation.numel() != num_records:
            raise ValueError(
                "generation and routed index metadata must address the same "
                "number of records."
            )

        target_device = self._conditional_update_device()
        if isinstance(patch, TensorDictBase):
            if patch.batch_size != leading_shape:
                raise ValueError(
                    "The patch batch size must match the routed index metadata."
                )
            flat_patch = patch.reshape(-1)
            if target_device is not None:
                flat_patch = flat_patch.to(target_device)
        else:
            flat_patch = {}
            for key, value in patch.items():
                if value.shape[: len(leading_shape)] != leading_shape:
                    raise ValueError(
                        f"Patch entry {key!r} must start with shape "
                        f"{tuple(leading_shape)}, got {tuple(value.shape)}."
                    )
                value = value.reshape(num_records, *value.shape[len(leading_shape) :])
                if target_device is not None:
                    value = value.to(target_device)
                flat_patch[key] = value

        updated = torch.zeros(
            num_records, dtype=torch.bool, device=flat_buffer_ids.device
        )
        version_rejected = (
            torch.zeros_like(updated) if version_key is not None else None
        )
        # One stable sort groups the records by member and keeps their
        # submission order within each group.
        order = flat_buffer_ids.argsort(stable=True)
        counts = torch.bincount(flat_buffer_ids, minlength=len(self._rbs)).tolist()
        with self._replay_lock, self._write_lock:
            start = 0
            for member_buffer, count in zip(self._rbs, counts):
                if not count:
                    continue
                rows = order[start : start + count]
                start += count
                member_patch = (
                    flat_patch[rows]
                    if isinstance(flat_patch, TensorDictBase)
                    else {
                        key: value[rows.to(value.device)]
                        for key, value in flat_patch.items()
                    }
                )
                member_version = version
                if isinstance(version, torch.Tensor) and version.numel() > 1:
                    member_version = version.reshape(
                        num_records, *version.shape[len(leading_shape) :]
                    )[rows.to(version.device)]
                result = member_buffer.update_if_present(
                    index=flat_local_index[rows.to(flat_local_index.device)],
                    generation=flat_generation[rows.to(flat_generation.device)],
                    patch=member_patch,
                    version_key=version_key,
                    version=member_version,
                    require_newer=require_newer,
                )
                updated[rows] = result.updated.to(updated.device)
                if version_rejected is not None:
                    version_rejected[rows] = result.version_rejected.to(
                        version_rejected.device
                    )
        return ConditionalUpdateResult(
            updated=updated.reshape(leading_shape),
            version_rejected=(
                version_rejected.reshape(leading_shape)
                if version_rejected is not None
                else None
            ),
            batch_size=leading_shape,
        )

    def end_streams(
        self,
        *,
        end_key: NestedKey = ("next", "done"),
        terminated_key: NestedKey | None = ("next", "terminated"),
        truncated_key: NestedKey | None = ("next", "truncated"),
    ) -> None:
        """Close each member's current stream before restarted producers append.

        Each member must hold one chronological stream in a one-dimensional
        tensor storage, with a generation-tracking round-robin writer and a
        :class:`SliceSampler` (including :class:`StreamingSliceSampler`) configured
        to read ``end_key``. A sampler with a ``traj_key`` is accepted only when
        every record of the member carries the same trajectory id (a per-stream
        key such as the environment index); with distinct ids, restarted
        producers may reuse old ones and the call raises.

        Pending samples and conditional updates finish before tail selection.
        Tail patches keep write counts and slot generations unchanged. Existing
        terminal/truncated flags are preserved; unfinished tails become done and,
        when the field exists, truncated. Uniform boundary caches and unfinished
        fresh windows are reset. Completed fresh windows and previously prefetched
        results retain their order; those results precede this operation.

        Quiesce collection before calling this method and until it returns.
        Empty members are skipped. Unsupported members are rejected before any
        tail is changed.

        Keyword Args:
            end_key (NestedKey, optional): Stored boolean boundary field and
                sampler end key. Defaults to ``("next", "done")``.
            terminated_key (NestedKey or None, optional): Stored terminal field,
                never modified. Missing fields are treated as false. Defaults
                to ``("next", "terminated")``.
            truncated_key (NestedKey or None, optional): Existing boolean field
                to set for unfinished nonterminal tails. ``None`` disables this
                patch. Defaults to ``("next", "truncated")``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.data import (
            ...     LazyTensorStorage, ReplayBufferEnsemble, SliceSampler,
            ...     TensorDictReplayBuffer, TensorDictRoundRobinWriter,
            ... )
            >>> member = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(8), sampler=SliceSampler(slice_len=2, end_key=("next", "done")),
            ...     writer=TensorDictRoundRobinWriter(track_generations=True),
            ... )
            >>> _ = member.extend(TensorDict({("next", "done"): torch.zeros(3, 1, dtype=torch.bool)}, [3]))
            >>> replay = ReplayBufferEnsemble(member)
            >>> replay.end_streams()
            >>> member[:]["next", "done"].flatten().tolist()
            [False, False, True]
        """
        end_key = unravel_key(end_key)
        terminated_key = (
            unravel_key(terminated_key) if terminated_key is not None else None
        )
        truncated_key = (
            unravel_key(truncated_key) if truncated_key is not None else None
        )
        keys = [
            key for key in (end_key, terminated_key, truncated_key) if key is not None
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("end, terminated and truncated keys must be distinct.")
        with self._futures_lock, ExitStack() as locks:
            self._synchronize_futures_locked()
            for member in self._rbs:
                locks.enter_context(member._futures_lock)
                member._synchronize_futures_locked()
            locks.enter_context(self._replay_lock)
            locks.enter_context(self._write_lock)
            for member in self._rbs:
                locks.enter_context(member._replay_lock)
                locks.enter_context(member._write_lock)
            patches = []
            for member in self._rbs:
                storage, writer, sampler = member.storage, member.writer, member.sampler
                if (
                    storage.ndim != 1
                    or not getattr(storage, "supports_conditional_update", False)
                    or not isinstance(writer, RoundRobinWriter)
                    or not writer.tracks_generations
                    or not isinstance(sampler, SliceSampler)
                    or end_key not in (sampler.end_keys or [sampler.end_key])
                ):
                    raise RuntimeError(
                        "end_streams requires one-dimensional conditional-update storage, "
                        "generation-tracking round-robin writers and end-key slice samplers."
                    )
                if not len(member):
                    continue
                if sampler.traj_key is not None:
                    # One trajectory id per member (a per-stream key such as the
                    # environment index) keeps the stream a single trajectory across
                    # restarts. Distinct ids may be reused by restarted producers,
                    # which a tail patch cannot separate.
                    data = storage._storage
                    trajectory = (
                        data.get(sampler.traj_key, default=None)
                        if is_tensor_collection(data)
                        else None
                    )
                    if trajectory is None or not bool(
                        (trajectory[: len(member)] == trajectory[0]).all()
                    ):
                        raise RuntimeError(
                            "end_streams supports trajectory-keyed slice samplers only "
                            "when every record of a member carries the same trajectory "
                            "id (a per-stream key such as the environment index); "
                            "restarted producers may otherwise reuse old ids."
                        )
                index = torch.tensor(
                    [(int(writer._cursor) - 1) % len(member)], device=storage.device
                )
                last = storage.get(index)
                done = last.get(end_key)
                if done.dtype != torch.bool:
                    raise TypeError("end_streams requires a boolean end_key field.")
                patch = {end_key: torch.ones_like(done)}
                if truncated_key is not None and truncated_key in last.keys(True):
                    terminated = (
                        last.get(terminated_key, None)
                        if terminated_key is not None
                        else None
                    )
                    unfinished = ~done
                    if terminated is not None:
                        unfinished = unfinished & ~terminated
                    patch[truncated_key] = last.get(truncated_key) | unfinished
                patches.append((member, index, writer.generations_of(index), patch))
            for member, index, generation, patch in patches:
                member.update_if_present(
                    index=index, generation=generation, patch=patch
                )
                member.sampler._end_stream(index, storage=member.storage)

    def stats(self) -> dict[str, int | float | bool]:
        """Returns aggregate scalar statistics across ensemble members."""
        if not self.initialized:
            return {
                "size": 0,
                "write_count": 0,
                "prefetch_queue_size": 0,
                "initialized": False,
                "num_buffers": len(self._init_storage._storages),
            }
        with self._replay_lock:
            size = sum(len(storage) for storage in self._storage._storages)
            capacity = sum(storage.max_size for storage in self._storage._storages)
            write_count = sum(
                getattr(writer, "_write_count", 0) for writer in self._writer._writers
            )
            prefetch_queue_size = len(self._prefetch_queue)
        return {
            "size": int(size),
            "write_count": int(write_count),
            "prefetch_queue_size": int(prefetch_queue_size),
            "initialized": True,
            "capacity": int(capacity),
            "utilization": float(size) / capacity if capacity else 0.0,
            "num_buffers": len(self._storage._storages),
        }

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
            p = (
                self._sampler._p[index]
                if isinstance(self._sampler._p, torch.Tensor)
                else self._sampler._p
            )
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
            p = (
                self._sampler._p[index]
                if isinstance(self._sampler._p, torch.Tensor)
                else self._sampler._p
            )

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
