# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import textwrap
from collections.abc import Callable, Mapping, Sequence
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


from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    SamplerEnsemble,
    SliceSampler,
)
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


from torchrl.data.replay_buffers.utils import _reduce

from .base import _storage_index, ConditionalUpdateResult, ReplayBuffer


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
        routing_mode (str, optional): how the routing key is read. "id"
            (default) expects one member id per record. "mask" expects a
            boolean mask with one column per member, and writes each record to
            every member marked True. HIL-SERL uses this to store intervened
            steps in both the demo and the online buffer. The mask is not
            stored. A member that receives only part of a trajectory has gaps
            in it; if it is sampled in slices, use a SliceSampler with
            fragmented=True so slices never cross a gap.
        fractions (sequence of float, optional): the share of each batch
            taken from each member, one value per member, summing to 1. The
            batch is then flat, one row per record, ordered by member, and
            each row records its member id and its index in that member.
            Every batch gets the same counts. If a member cannot provide its
            share, sampling fails instead of taking more from the others. The
            members' outputs are concatenated, so collate_fn is not used.
            With prefetch, a change of fractions applies after the batches
            already prefetched. Cannot be combined with p, sample_from_all or
            num_buffer_sampled. See SamplerEnsemble.
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
        routing_mode: Literal["id", "mask"] = "id",
        fractions: Sequence[float] | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        shared: bool = False,
        **kwargs,
    ):

        if routing_key is not None and routing_dim is not None:
            raise ValueError("routing_key and routing_dim are mutually exclusive.")
        if routing_mode not in ("id", "mask"):
            raise ValueError(
                f"routing_mode must be 'id' or 'mask', got {routing_mode!r}."
            )
        if routing_mode == "mask" and routing_key is None:
            raise ValueError("routing_mode='mask' requires a routing_key.")
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
                fractions=fractions,
            )
            writers = WriterEnsemble(*[rb._writer for rb in rbs])
            if collate_fns is None:
                collate_fns = [rb._collate_fn for rb in rbs]
        else:
            if fractions is not None:
                samplers.fractions = fractions
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
        self.routing_mode = routing_mode
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
        if self.routing_mode == "mask":
            mask = flat_data.get(self.routing_key)
            num_members = len(self._storage._storages)
            if mask.dtype != torch.bool:
                raise TypeError(
                    f"routing_mode='mask' expects a boolean mask under "
                    f"{self.routing_key!r}, got {mask.dtype}."
                )
            if mask.shape != (*flat_data.batch_size, num_members):
                raise ValueError(
                    f"routing_mode='mask' expects a [*B, {num_members}] mask under "
                    f"{self.routing_key!r}, got shape {tuple(mask.shape)} for "
                    f"batch size {tuple(flat_data.batch_size)}."
                )
            return flat_data, mask
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
                torch.arange(data.shape[routing_dim], device=data.device or "cpu")
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
        """Writes a batch to the members and returns where each record went.

        The result has one row per record: the member id, the index in that
        member, and the slot generation when every member tracks generations.

        In mask mode the result has one row per record and one column per
        member, so a row holds the address of every copy of that record.
        "present" marks the copies that were written. Missing copies have
        index -1, and update_if_present and update_priority skip them. A patch
        given to update_if_present with these addresses applies to every copy
        of the record. Filter on "present" before using an index directly,
        because -1 is a valid tensor index.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.data import (
            ...     LazyTensorStorage, ReplayBufferEnsemble, TensorDictReplayBuffer,
            ...     TensorDictRoundRobinWriter,
            ... )
            >>> demo, online = (
            ...     TensorDictReplayBuffer(
            ...         storage=LazyTensorStorage(8),
            ...         writer=TensorDictRoundRobinWriter(track_generations=True),
            ...     )
            ...     for _ in range(2)
            ... )
            >>> rb = ReplayBufferEnsemble(demo, online, routing_key="route", routing_mode="mask")
            >>> intervention = torch.tensor([True, False, True])
            >>> data = TensorDict(
            ...     {"reward": torch.zeros(3), "route": torch.stack([intervention, torch.ones(3, dtype=torch.bool)], -1)},
            ...     [3],
            ... )
            >>> addresses = rb.extend(data)
            >>> addresses["index"].tolist()
            [[0, 0], [-1, 1], [1, 2]]
            >>> result = rb.update_if_present(
            ...     index=addresses.select("buffer_ids", "index", "present"),
            ...     generation=addresses["index_generation"],
            ...     patch={"reward": torch.tensor([1.0, 0.0, 1.0])},
            ... )
            >>> demo[:]["reward"].tolist(), online[:]["reward"].tolist()
            ([1.0, 1.0], [1.0, 0.0, 1.0])
        """
        if update_priority is not None:
            raise NotImplementedError(
                "update_priority is not supported by routed ensemble writes."
            )
        if self.routing_mode == "mask":
            return self._extend_mask(data)
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
                member_index, generation = self._write_member(
                    member_buffer, flat_data[positions], buffer_ids.device
                )
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

                if generation is not None:
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
        self._notify_replay_state_change()
        return metadata

    def _write_member(
        self, member_buffer: ReplayBuffer, member_data: TensorDictBase, device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        member_index = member_buffer.extend(member_data)
        if isinstance(member_index, tuple):
            member_index = torch.stack(member_index, -1)
        else:
            member_index = torch.as_tensor(member_index, device=device)
        if member_index.ndim == 0:
            member_index = member_index.unsqueeze(0)
        member_index = member_index.to(device)
        if not self._writer.tracks_generations:
            return member_index, None
        generation = member_buffer.writer.generations_of(member_index)
        slots = (
            member_index[..., 0] if member_buffer.storage.ndim > 1 else member_index
        ).reshape(-1)
        max_size = getattr(member_buffer.storage, "max_size", None)
        may_wrap = (
            not isinstance(member_buffer.writer, RoundRobinWriter)
            or max_size is None
            or slots.numel() > max_size
        )
        if may_wrap and slots.unique().numel() != slots.numel():
            later_occurrences = torch.empty_like(slots)
            seen = {}
            for position in range(slots.numel() - 1, -1, -1):
                slot = int(slots[position].cpu())
                later_occurrences[position] = seen.get(slot, 0)
                seen[slot] = seen.get(slot, 0) + 1
            generation = generation.reshape(-1) - later_occurrences.to(
                generation.device
            )
        return member_index, generation.reshape(-1).to(device)

    def _extend_mask(self, data: TensorDictBase) -> TensorDictBase:
        flat_data, mask = self._route(data)
        flat_data = flat_data.exclude(self.routing_key)
        num_records, num_members = mask.shape
        device = mask.device
        local_indices = None
        local_generations = None
        counts = mask.sum(0).tolist()
        with self._replay_lock, self._write_lock:
            for member_id, member_buffer in enumerate(self._rbs):
                if not counts[member_id]:
                    continue
                if counts[member_id] == num_records:
                    positions = slice(None)
                    member_data = flat_data
                else:
                    positions = mask[:, member_id].nonzero().flatten()
                    member_data = flat_data[positions]
                member_index, generation = self._write_member(
                    member_buffer, member_data, device
                )
                if local_indices is None:
                    local_indices = torch.full(
                        (num_records, num_members, *member_index.shape[1:]),
                        -1,
                        dtype=member_index.dtype,
                        device=device,
                    )
                elif member_index.shape[1:] != local_indices.shape[2:]:
                    raise RuntimeError(
                        "All routed replay-buffer members must use compatible "
                        "index shapes."
                    )
                local_indices[positions, member_id] = member_index
                if generation is not None:
                    if local_generations is None:
                        local_generations = torch.full(
                            (num_records, num_members),
                            -1,
                            dtype=generation.dtype,
                            device=device,
                        )
                    local_generations[positions, member_id] = generation
        if local_indices is None:
            local_indices = torch.full(
                (num_records, num_members), -1, dtype=torch.long, device=device
            )
        if local_generations is None and self._writer.tracks_generations:
            local_generations = torch.full(
                (num_records, num_members), -1, dtype=torch.long, device=device
            )
        metadata = TensorDict(
            {
                "buffer_ids": torch.arange(num_members, device=device).expand(
                    num_records, num_members
                ),
                "index": local_indices,
                "present": mask,
            },
            batch_size=[num_records, num_members],
        )
        if local_generations is not None:
            metadata.set("index_generation", local_generations)
        self._notify_replay_state_change()
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
        present = index.get("present", None)
        if present is not None:
            return self._update_present_copies(
                index=index,
                present=present,
                generation=generation,
                patch=patch,
                version_key=version_key,
                version=version,
                require_newer=require_newer,
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
        with self._replay_lock, self._write_lock:
            for member_buffer, rows in self._group_rows(flat_buffer_ids):
                member_patch = (
                    flat_patch[rows.to(flat_patch.device or "cpu")]
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

    def _update_present_copies(
        self,
        *,
        index: TensorDictBase,
        present: torch.Tensor,
        generation: torch.Tensor,
        patch: Mapping[NestedKey, torch.Tensor] | TensorDictBase,
        version_key: NestedKey | None,
        version: int | torch.Tensor | None,
        require_newer: bool,
    ) -> ConditionalUpdateResult:
        leading_shape = present.shape
        record_shape = leading_shape[:-1]

        def broadcast(value):
            if value.shape[: len(record_shape)] != record_shape:
                raise ValueError(
                    f"Patch entries must start with the record shape "
                    f"{tuple(record_shape)}, got {tuple(value.shape)}. Pass a "
                    f"TensorDict with batch size {tuple(leading_shape)} to patch "
                    "each copy separately."
                )
            value = value.unsqueeze(len(record_shape))
            return value.expand(*leading_shape, *value.shape[len(leading_shape) :])

        keep = present.reshape(-1)
        if isinstance(patch, TensorDictBase):
            if patch.batch_size == record_shape:
                patch = patch.unsqueeze(len(record_shape)).expand(leading_shape)
            flat_patch = patch.reshape(-1)[keep]
        else:
            flat_patch = {}
            for key, value in patch.items():
                value = broadcast(value)
                flat_patch[key] = value.reshape(
                    keep.numel(), *value.shape[len(leading_shape) :]
                )[keep]
        if isinstance(version, torch.Tensor) and version.numel() > 1:
            version = broadcast(version)
            version = version.reshape(
                keep.numel(), *version.shape[len(leading_shape) :]
            )[keep]
        flat_index = index.exclude("present").reshape(-1)[keep]
        result = self.update_if_present(
            index=flat_index,
            generation=generation.reshape(-1)[keep],
            patch=flat_patch,
            version_key=version_key,
            version=version,
            require_newer=require_newer,
        )
        updated = torch.zeros(keep.numel(), dtype=torch.bool, device=keep.device)
        updated[keep] = result.updated.to(keep.device)
        version_rejected = None
        if result.version_rejected is not None:
            version_rejected = torch.zeros_like(updated)
            version_rejected[keep] = result.version_rejected.to(keep.device)
            version_rejected = version_rejected.reshape(leading_shape)
        return ConditionalUpdateResult(
            updated=updated.reshape(leading_shape),
            version_rejected=version_rejected,
            batch_size=leading_shape,
        )

    def _group_rows(self, flat_buffer_ids: torch.Tensor):
        """Yields each member with the rows that belong to it, in their original order."""
        order = flat_buffer_ids.argsort(stable=True)
        counts = torch.bincount(flat_buffer_ids, minlength=len(self._rbs)).tolist()
        start = 0
        for member_buffer, count in zip(self._rbs, counts):
            if count:
                yield member_buffer, order[start : start + count]
            start += count

    def _flat_routed_index(
        self, index: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        if not isinstance(index, TensorDictBase):
            raise TypeError(
                "ReplayBufferEnsemble priority updates require routed index "
                "metadata with 'buffer_ids' and 'index' entries."
            )
        buffer_ids = index.get("buffer_ids")
        local_index = index.get("index")
        leading_shape = buffer_ids.shape
        if local_index.shape[: len(leading_shape)] != leading_shape:
            raise ValueError(
                "Member-local indices must start with the buffer-id shape, got "
                f"{tuple(local_index.shape)} and {tuple(leading_shape)}."
            )
        num_records = buffer_ids.numel()
        return (
            buffer_ids.reshape(-1).to(torch.long),
            local_index.reshape(num_records, *local_index.shape[len(leading_shape) :]),
            leading_shape,
        )

    def update_priority(
        self,
        index: TensorDictBase,
        priority: float | torch.Tensor,
    ) -> None:
        """Sends new priorities to the members that hold the records.

        Members without a prioritized sampler are skipped. With fractions set,
        each member's share of a batch is fixed, so priorities only change
        sampling inside a member.

        Args:
            index (TensorDictBase): record addresses, such as the "index"
                entry of a sampled batch or the result of extend. Rows marked
                False in an optional "present" entry are skipped.
            priority (float or torch.Tensor): the new priorities, one value
                for all records or one per record.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.data import (
            ...     LazyTensorStorage, PrioritizedSampler, ReplayBufferEnsemble,
            ...     TensorDictReplayBuffer,
            ... )
            >>> demo = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(8), sampler=PrioritizedSampler(8, alpha=1.0, beta=0.5)
            ... )
            >>> online = TensorDictReplayBuffer(storage=LazyTensorStorage(8))
            >>> rb = ReplayBufferEnsemble(demo, online, routing_key="route")
            >>> _ = rb.extend(TensorDict({"obs": torch.arange(4), "route": torch.tensor([0, 0, 1, 1])}, [4]))
            >>> index = TensorDict({"buffer_ids": torch.tensor([0]), "index": torch.tensor([1])}, [1])
            >>> rb.update_priority(index, torch.tensor([1e4]))
            >>> (demo.sample(100)["obs"] == 1).float().mean() > 0.9
            tensor(True)
        """
        flat_buffer_ids, flat_local_index, leading_shape = self._flat_routed_index(
            index
        )
        priority = torch.as_tensor(priority)
        if priority.ndim:
            if priority.shape[: len(leading_shape)] != leading_shape:
                raise ValueError(
                    f"priority must have the leading shape {tuple(leading_shape)} "
                    f"of the routed index, got {tuple(priority.shape)}."
                )
            priority = priority.reshape(flat_buffer_ids.numel(), -1).squeeze(-1)
        else:
            priority = priority.expand(flat_buffer_ids.numel())
        present = index.get("present", None)
        if present is not None:
            keep = present.reshape(-1)
            flat_buffer_ids = flat_buffer_ids[keep]
            flat_local_index = flat_local_index[keep]
            priority = priority[keep]
        for member_buffer, rows in self._group_rows(flat_buffer_ids):
            if not hasattr(member_buffer.sampler, "update_priority"):
                continue
            member_buffer.update_priority(
                flat_local_index[rows.to(flat_local_index.device)],
                priority[rows.to(priority.device)],
            )

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        """Updates each member's priorities from a sampled batch.

        Each row goes to the member it was sampled from, which reads its own
        priority key. Plain ReplayBuffer members read "td_error". Members
        without a prioritized sampler are skipped. Works with flat batches and
        with the default stacked layout.

        Args:
            data (TensorDictBase): a batch from sample, with the priorities
                the loss wrote into it (for instance "td_error").

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.data import (
            ...     LazyTensorStorage, PrioritizedSampler, ReplayBufferEnsemble,
            ...     TensorDictReplayBuffer,
            ... )
            >>> demo = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(8), sampler=PrioritizedSampler(8, alpha=0.6, beta=0.4)
            ... )
            >>> online = TensorDictReplayBuffer(storage=LazyTensorStorage(8))
            >>> rb = ReplayBufferEnsemble(demo, online, routing_key="route", sample_from_all=True)
            >>> _ = rb.extend(TensorDict({"obs": torch.arange(8), "route": torch.arange(8) // 4}, [8]))
            >>> batch = rb.sample(4)
            >>> _ = batch.set("td_error", torch.rand(batch.shape))
            >>> rb.update_tensordict_priority(batch)
        """
        index = data.get("index", None)
        if not isinstance(index, TensorDictBase):
            raise TypeError(
                "update_tensordict_priority expects a batch sampled from the "
                "ensemble, with an 'index' entry holding 'buffer_ids' and 'index'."
            )
        flat_buffer_ids, flat_local_index, leading_shape = self._flat_routed_index(
            index
        )
        num_records = flat_buffer_ids.numel()

        def flat_entry(key):
            value = data.get(key, None)
            if value is None:
                return None
            return value.reshape(num_records, *value.shape[len(leading_shape) :])

        extras = {
            key: value
            for key in ("anchor_index", "validity_mask")
            if (value := flat_entry(key)) is not None
        }
        for member_buffer, rows in self._group_rows(flat_buffer_ids):
            sampler = member_buffer.sampler
            if not isinstance(sampler, PrioritizedSampler):
                continue
            priority_key = getattr(member_buffer, "priority_key", "td_error")
            priority = flat_entry(priority_key)
            update = getattr(member_buffer, "update_tensordict_priority", None)
            if update is None:
                if priority is not None:
                    member_buffer.update_priority(
                        flat_local_index[rows],
                        _reduce(
                            priority[rows].reshape(rows.numel(), -1),
                            sampler.reduction,
                            dim=1,
                        ),
                    )
                continue
            entries = {"index": flat_local_index[rows]}
            if priority is not None:
                entries[priority_key] = priority[rows]
            for key, value in extras.items():
                entries[key] = value[rows]
            update(TensorDict(entries, batch_size=[rows.numel()]))

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
                "storage_size": 0,
                "sampleable_size": 0,
                "write_count": 0,
                "sample_calls": self._counter_value(self._sample_call_count_value),
                "samples_returned": self._counter_value(self._sampled_item_count_value),
                "prefetch_queue_size": 0,
                "initialized": False,
                "num_buffers": len(self._init_storage._storages),
            }
        with self._replay_lock:
            storage_size = sum(len(storage) for storage in self._storage._storages)
            sampleable_size = sum(len(member) for member in self._rbs)
            capacity = sum(storage.max_size for storage in self._storage._storages)
            write_count = sum(
                getattr(writer, "_write_count", 0) for writer in self._writer._writers
            )
            prefetch_queue_size = len(self._prefetch_queue)
        return {
            "size": int(sampleable_size),
            "storage_size": int(storage_size),
            "sampleable_size": int(sampleable_size),
            "write_count": int(write_count),
            "sample_calls": self._counter_value(self._sample_call_count_value),
            "samples_returned": self._counter_value(self._sampled_item_count_value),
            "prefetch_queue_size": int(prefetch_queue_size),
            "initialized": True,
            "capacity": int(capacity),
            "utilization": float(storage_size) / capacity if capacity else 0.0,
            "num_buffers": len(self._storage._storages),
        }

    @property
    def fractions(self) -> torch.Tensor | None:
        """The share of each batch taken from each member, or None.

        Assign a new value between batches to change the mix.
        """
        return self._sampler.fractions

    @fractions.setter
    def fractions(self, value: Sequence[float] | torch.Tensor | None) -> None:
        self._sampler.fractions = value

    def _sample_flat(self, batch_size: int) -> tuple[TensorDictBase, TensorDictBase]:
        with self._replay_lock, self._write_lock:
            index, info = self._sampler.sample(self._storage, batch_size)
            local_index = index.get("index")
            transforms = self._storage._transforms
            parts = []
            generations = []
            start = 0
            for member_id, count in enumerate(self._sampler.quota_counts(batch_size)):
                if not count:
                    continue
                member = self._rbs[member_id]
                member_index = local_index[start : start + count]
                start += count
                storage_index = (
                    member_index.unbind(-1)
                    if member.storage.ndim > 1 and member_index.ndim == 2
                    else member_index
                )
                data = member.storage.get(_storage_index(storage_index, member.storage))
                if transforms is not None and transforms[member_id] is not None:
                    data = transforms[member_id](data)
                parts.append(self._collate_fns[member_id](data))
                if self._writer.tracks_generations:
                    generations.append(member.writer.generations_of(member_index))
            info.set("index", index)
            if generations:
                info.set("index_generation", torch.cat(generations))
        data = torch.cat(parts)
        is_td = is_tensor_collection(data)
        if self._transform is not None and len(self._transform):
            with (
                data.unlock_() if is_td else contextlib.nullcontext()
            ), _set_dispatch_td_nn_modules(is_td):
                data = self._transform(data)
        if is_td:
            data.update(info)
        return data, info

    def _sample(self, *args, **kwargs):
        if self._sampler.fractions is not None:
            return self._sample_flat(*args, **kwargs)
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
            return ReplayBufferEnsemble(
                *rbs,
                transform=self._transform,
                batch_size=self._batch_size,
                collate_fn=self._collate_fn_val,
                collate_fns=_collate_fns,
                **self._sampling_strategy(index),
            )

        try:
            samplers = self._sampler[index]
            writers = self._writer[index]
            storages = self._storage[index]
            if isinstance(index, torch.Tensor):
                _collate_fns = [self._collate_fns[i] for i in index.tolist()]
            else:
                _collate_fns = self._collate_fns[index]
        except IndexError:
            raise IndexError(self._INDEX_ERROR.format(type(index)))

        if samplers.fractions is not None:
            return ReplayBufferEnsemble(
                samplers=samplers,
                writers=writers,
                storages=storages,
                transform=self._transform,
                batch_size=self._batch_size,
                collate_fn=self._collate_fn_val,
                collate_fns=_collate_fns,
            )
        return ReplayBufferEnsemble(
            samplers=samplers,
            writers=writers,
            storages=storages,
            transform=self._transform,
            batch_size=self._batch_size,
            collate_fn=self._collate_fn_val,
            collate_fns=_collate_fns,
            **self._sampling_strategy(index),
        )

    def _sampling_strategy(self, index) -> dict[str, Any]:
        if self._sampler.fractions is not None:
            fractions = (
                self._sampler.fractions[torch.as_tensor(index)]
                if isinstance(index, list)
                else self._sampler.fractions[index]
            )
            if not float(fractions.sum()):
                raise ValueError("The selected members all have a fraction of 0.")
            return {"fractions": fractions / fractions.sum()}
        p = (
            self._sampler._p[index]
            if isinstance(self._sampler._p, torch.Tensor)
            else self._sampler._p
        )
        return {
            "sample_from_all": self._sampler.sample_from_all,
            "num_buffer_sampled": self._sampler.num_buffer_sampled,
            "p": p,
        }

    def __len__(self):
        return len(self._storage)

    def __repr__(self):
        storages = textwrap.indent(f"storages={self._storage}", " " * 4)
        writers = textwrap.indent(f"writers={self._writer}", " " * 4)
        samplers = textwrap.indent(f"samplers={self._sampler}", " " * 4)
        return f"ReplayBufferEnsemble(\n{storages}, \n{samplers}, \n{writers}, \nbatch_size={self._batch_size}, \ntransform={self._transform}, \ncollate_fn={self._collate_fn_val})"
