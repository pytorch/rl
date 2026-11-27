# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import textwrap
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.storages import StorageEnsemble

from .base import Sampler

_PRIORITY_WEIGHT_KEY = "priority_weight"


class SamplerEnsemble(Sampler):
    """An ensemble of samplers.

    This class is designed to work with :class:`~torchrl.data.replay_buffers.replay_buffers.ReplayBufferEnsemble`.
    It contains the samplers as well as the sampling strategy hyperparameters.

    Args:
        samplers (sequence of Sampler): the samplers to make the composite sampler.

    Keyword Args:
        p (list, tensor of probabilities, or ``"sampleable"``, optional): if
            provided, indicates the weights of each dataset during sampling.
            ``"sampleable"`` recomputes weights from the number of records or
            valid slice windows currently available in each member and excludes
            members that cannot provide a batch.
        sample_from_all (bool, optional): if ``True``, each dataset will be sampled
            from. This is not compatible with the ``p`` argument. Defaults to ``False``.
        num_buffer_sampled (int, optional): the number of buffers to sample.
            if ``sample_from_all=True``, this has no effect, as it defaults to the
            number of buffers. If ``sample_from_all=False``, buffers will be
            sampled according to the probabilities ``p``.
        fractions (sequence of float or torch.Tensor, optional): the share of each batch
            taken from each member, one value per member, summing to 1. Every
            batch gets the same counts, rounded so they add up to the batch
            size. The sampled index is flat and ordered by member. With slice
            samplers the counts are in slices, so the batch size must be a
            multiple of the slice length. An empty member makes sampling
            fail instead of taking more from the others, and a member with
            fewer records than its share repeats records, even with a
            without-replacement sampler. Cannot be combined with p, sample_from_all or num_buffer_sampled.
            Can be changed between batches. Defaults to None.

    .. warning::
      The indices provided in the info dictionary are placed in a :class:`~tensordict.TensorDict` with
      keys ``index`` and ``buffer_ids`` that allow the upper :class:`~torchrl.data.ReplayBufferEnsemble`
      and :class:`~torchrl.data.StorageEnsemble` objects to retrieve the data.
      This format is different from with other samplers which usually return indices
      as regular tensors.

    .. seealso:: :class:`~torchrl.trainers.algorithms.configs.data.SamplerEnsembleConfig`

    """

    _fractions = None

    @property
    def requires_shared_state(self) -> bool:
        return any(sampler.requires_shared_state for sampler in self._samplers)

    def __init__(
        self,
        *samplers,
        p: list[float] | torch.Tensor | Literal["sampleable"] | None = None,
        sample_from_all: bool = False,
        num_buffer_sampled: int | None = None,
        fractions: Sequence[float] | torch.Tensor | None = None,
    ):
        self._rng_private = None
        self._samplers = samplers
        self.sample_from_all = sample_from_all
        if sample_from_all and p is not None:
            raise RuntimeError(
                "Cannot pass both `p` argument and `sample_from_all=True`."
            )
        if fractions is not None and (
            p is not None or sample_from_all or num_buffer_sampled is not None
        ):
            raise ValueError(
                "fractions is exclusive with p, sample_from_all and num_buffer_sampled."
            )
        self.p = p
        self.num_buffer_sampled = num_buffer_sampled
        self.fractions = fractions

    @property
    def _rng(self):
        return self._rng_private

    @_rng.setter
    def _rng(self, value):
        self._rng_private = value
        for sampler in self._samplers:
            sampler._rng = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if isinstance(value, str):
            if value != "sampleable":
                raise ValueError(
                    "The only dynamic SamplerEnsemble probability is 'sampleable'."
                )
            self._p = value
            return
        if not isinstance(value, torch.Tensor) and value is not None:
            value = torch.tensor(value)
        if value is not None:
            value = value / value.sum().clamp_min(1e-6)
        self._p = value

    @property
    def fractions(self) -> torch.Tensor | None:
        """The share of each batch taken from each member, or None.

        Returns a copy. Assign a new value to change the shares.
        """
        return self._fractions.clone() if self._fractions is not None else None

    @fractions.setter
    def fractions(self, value: Sequence[float] | torch.Tensor | None) -> None:
        self._quota_cache = {}
        if value is None:
            self._fractions = None
            return
        num_buffer_sampled = self.__dict__.get("_num_buffer_sampled")
        if (
            getattr(self, "_p", None) is not None
            or self.sample_from_all
            or num_buffer_sampled not in (None, len(self._samplers))
        ):
            raise ValueError(
                "fractions is exclusive with p, sample_from_all and num_buffer_sampled."
            )
        value = torch.as_tensor(value, dtype=torch.double).flatten()
        if value.numel() != len(self._samplers):
            raise ValueError(
                f"fractions must contain one fraction per member: got "
                f"{value.numel()} for {len(self._samplers)} members."
            )
        if (value < 0).any():
            raise ValueError("fractions must be non-negative.")
        if not math.isclose(float(value.sum()), 1.0, abs_tol=1e-6):
            raise ValueError(f"fractions must sum to 1, got {float(value.sum())}.")
        self._fractions = value

    def _slice_unit(self) -> int:
        units = set()
        for member_id, sampler in enumerate(self._samplers):
            if not hasattr(sampler, "slice_len"):
                units.add(1)
                continue
            if getattr(sampler, "output_layout", "flat") != "flat":
                raise ValueError(
                    f"fractions requires flat sampler output; member {member_id} "
                    "uses output_layout='batch_time'."
                )
            if sampler.slice_len is None:
                raise ValueError(
                    f"fractions requires slice samplers to set slice_len; member "
                    f"{member_id} sets num_slices instead."
                )
            if not getattr(sampler, "strict_length", True):
                raise ValueError(
                    f"fractions requires slice samplers with strict_length=True; "
                    f"member {member_id} can return short batches."
                )
            units.add(int(sampler.slice_len))
        if len(units) > 1:
            raise ValueError(
                "fractions requires every member to use the same slice length, "
                f"got {sorted(units)}."
            )
        return units.pop()

    def quota_counts(self, batch_size: int) -> list[int]:
        """Returns how many records each member gives to a batch.

        Args:
            batch_size (int): the batch size, in records.

        Returns:
            one count per member. The counts add up to the batch size.

        Examples:
            >>> from torchrl.data.replay_buffers.samplers import RandomSampler, SamplerEnsemble
            >>> sampler = SamplerEnsemble(RandomSampler(), RandomSampler(), fractions=[0.3, 0.7])
            >>> sampler.quota_counts(256)
            [77, 179]
        """
        if self._fractions is None:
            raise RuntimeError("quota_counts requires fractions to be set.")
        if batch_size <= 0:
            raise ValueError(f"The batch size must be positive, got {batch_size}.")
        unit = self._slice_unit()
        cached = self._quota_cache.get((batch_size, unit))
        if cached is None:
            cached = self._quota_cache[(batch_size, unit)] = self._compute_quota_counts(
                batch_size, unit
            )
        return list(cached)

    def _compute_quota_counts(self, batch_size: int, unit: int) -> list[int]:
        if batch_size % unit:
            raise ValueError(
                f"With fractions, the batch size ({batch_size}) must be a multiple "
                f"of the slice length ({unit})."
            )
        units = batch_size // unit
        raw = [fraction * units for fraction in self._fractions.tolist()]
        counts = [math.floor(value) for value in raw]
        remainder = units - sum(counts)
        order = sorted(
            range(len(counts)),
            key=lambda member_id: (counts[member_id] - raw[member_id], member_id),
        )
        for member_id in order[:remainder]:
            counts[member_id] += 1
        return [count * unit for count in counts]

    def _sample_fractions(self, storage, batch_size):
        counts = self.quota_counts(batch_size)
        blocked = [
            member_id
            for member_id, (count, sampler, member_storage) in enumerate(
                zip(counts, self._samplers, storage._storages)
            )
            if count and not sampler.can_sample(member_storage, count)
        ]
        if blocked:
            raise RuntimeError(
                f"Ensemble member(s) {blocked} cannot serve their share of a batch "
                f"of {batch_size} (counts {counts}). Fill them, or set their "
                "fraction to 0."
            )
        member_ids = []
        index_parts = []
        infos = []
        for member_id, count in enumerate(counts):
            if not count:
                continue
            member_index, member_info = self._sample_member(storage, member_id, count)
            member_ids.append(member_id)
            index_parts.append(member_index)
            infos.append(member_info)
        buffer_ids = torch.tensor(member_ids, dtype=torch.long)
        self._check_index_layouts(index_parts, buffer_ids)
        device = index_parts[0].device
        index_parts = [part.to(device) for part in index_parts]
        infos = [info.to(device) for info in infos]
        infos = self._fill_neutral_infos(infos, buffer_ids, index_parts)
        flat_buffer_ids = torch.empty(batch_size, dtype=torch.long, device=device)
        start = 0
        for member_id, part in zip(member_ids, index_parts):
            flat_buffer_ids[start : start + part.shape[0]] = member_id
            start += part.shape[0]
        index = TensorDict(
            {"index": torch.cat(index_parts), "buffer_ids": flat_buffer_ids},
            batch_size=[batch_size],
        )
        return index, torch.cat(infos)

    def _sample_member(self, storage, member_id, count):
        parts = []
        infos = []
        remaining = count
        while remaining:
            part, info = self._samplers[member_id].sample(
                storage._storages[member_id], remaining
            )
            if not isinstance(part, torch.Tensor):
                part = part[0] if len(part) == 1 else torch.stack(part, -1)
            if not part.shape[0] or part.shape[0] > remaining:
                raise RuntimeError(
                    f"Sampler {member_id} returned {part.shape[0]} records for a "
                    f"requested batch of {remaining}."
                )
            parts.append(part)
            part_info = TensorDict(batch_size=part.shape[:1])
            if info:
                if not isinstance(info, TensorDict):
                    info = TensorDict.from_dict(info, batch_dims=0)
                for key, value in info.items(True, True):
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim
                        and value.shape[0] == part.shape[0]
                    ):
                        part_info.set(key, value)
            infos.append(part_info)
            remaining -= part.shape[0]
        if len(parts) == 1:
            return parts[0], infos[0]
        return torch.cat(parts), torch.cat(infos)

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
        if not isinstance(storage, StorageEnsemble):
            raise TypeError("SamplerEnsemble requires a StorageEnsemble.")
        if self._fractions is not None:
            return self._sample_fractions(storage, batch_size)
        if batch_size % self.num_buffer_sampled > 0:
            raise ValueError("The batch size must be divisible by num_buffer_sampled.")
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
            if isinstance(self.p, str):
                counts = torch.tensor(
                    [
                        float(sampler._sampleable_count(member_storage, sub_batch_size))
                        for member_storage, sampler in zip(
                            storage._storages, self._samplers
                        )
                    ],
                    dtype=torch.float,
                    device=self._rng.device if self._rng is not None else "cpu",
                )
                if not counts.any():
                    raise RuntimeError(
                        "None of the replay-buffer ensemble members can be sampled."
                    )
                probabilities = counts / counts.sum()
                buffer_ids = torch.multinomial(
                    probabilities,
                    self.num_buffer_sampled,
                    True,
                    generator=self._rng,
                )
            elif self.p is None:
                buffer_ids = torch.randint(
                    len(self._samplers),
                    (self.num_buffer_sampled,),
                    generator=self._rng,
                    device=getattr(storage, "device", None),
                )
            else:
                buffer_ids = torch.multinomial(
                    self.p,
                    self.num_buffer_sampled,
                    True,
                    generator=self._rng,
                )
            samples = [None] * self.num_buffer_sampled
            infos = [None] * self.num_buffer_sampled
            for member_id in buffer_ids.unique(sorted=True).tolist():
                positions = (buffer_ids == member_id).nonzero().flatten()
                member_batch_size = sub_batch_size * positions.numel()
                member_samples = []
                member_infos = []
                remaining = member_batch_size
                while remaining:
                    member_sample, member_info = self._samplers[member_id].sample(
                        storage._storages[member_id], remaining
                    )
                    if not isinstance(member_sample, torch.Tensor):
                        member_sample = torch.stack(member_sample, -1)
                    if not member_sample.shape[0] or member_sample.shape[0] > remaining:
                        raise RuntimeError(
                            f"Sampler {member_id} returned {member_sample.shape[0]} "
                            f"records for a requested batch of {remaining}."
                        )
                    member_samples.append(member_sample)
                    member_infos.append(member_info)
                    remaining -= member_sample.shape[0]
                member_sample = torch.cat(member_samples)
                if len(member_infos) == 1:
                    member_info = member_infos[0]
                else:
                    member_info = {}
                    keys = set().union(*(info.keys() for info in member_infos))
                    part_sizes = [part.shape[0] for part in member_samples]
                    for key in keys:
                        values = [info.get(key) for info in member_infos]
                        if all(
                            isinstance(value, torch.Tensor)
                            and value.ndim
                            and value.shape[0] == part_size
                            for value, part_size in zip(values, part_sizes)
                        ):
                            member_info[key] = torch.cat(values)
                        else:
                            member_info[key] = next(
                                value for value in reversed(values) if value is not None
                            )
                sample_chunks = member_sample.split(sub_batch_size, dim=0)
                member_info = (
                    TensorDict.from_dict(member_info, batch_dims=0)
                    if member_info
                    else TensorDict()
                )
                for chunk_id, position in enumerate(positions.tolist()):
                    samples[position] = sample_chunks[chunk_id]
                    start = chunk_id * sub_batch_size
                    stop = start + sub_batch_size
                    info_chunk = member_info.clone(False)
                    for key, value in member_info.items(
                        include_nested=True, leaves_only=True
                    ):
                        if (
                            isinstance(value, torch.Tensor)
                            and value.ndim
                            and value.shape[0] == member_batch_size
                        ):
                            info_chunk.set(key, value[start:stop])
                    infos[position] = info_chunk
        samples = [
            sample if isinstance(sample, torch.Tensor) else torch.stack(sample, -1)
            for sample in samples
        ]
        self._check_index_layouts(samples, buffer_ids)
        index_parts = samples
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
        if not isinstance(infos, list):
            infos = [
                TensorDict.from_dict(info, batch_dims=samples.ndim - 1)
                if info
                else TensorDict()
                for info in infos
            ]
        infos = torch.stack(self._fill_neutral_infos(infos, buffer_ids, index_parts))
        return samples, infos

    @staticmethod
    def _check_index_layouts(samples, buffer_ids) -> None:
        layouts = {}
        for sample, member_id in zip(samples, buffer_ids.tolist()):
            layouts.setdefault(sample.ndim, set()).add(member_id)
        if len(layouts) > 1:
            details = ", ".join(
                f"member(s) {sorted(members)} return {ndim}-d indices"
                for ndim, members in sorted(layouts.items())
            )
            raise ValueError(
                "Ensemble members return incompatible index layouts: "
                f"{details}. Use samplers of the same family (all slice "
                "samplers, or none) across members."
            )

    @staticmethod
    def _fill_neutral_infos(infos, buffer_ids, index_parts):
        keys = [set(info.keys(True, True)) for info in infos]
        all_keys = set().union(*keys)
        if all(member_keys == all_keys for member_keys in keys):
            return infos
        member_ids = buffer_ids.tolist()
        for key in all_keys:
            missing = [
                member_id
                for member_id, member_keys in zip(member_ids, keys)
                if key not in member_keys
            ]
            if missing and key != _PRIORITY_WEIGHT_KEY:
                raise KeyError(
                    f"Ensemble members return different sampler info: {key!r} "
                    f"is missing from member(s) {sorted(set(missing))}. Only "
                    f"{_PRIORITY_WEIGHT_KEY!r} is filled for members that do not "
                    "provide it."
                )
        if _PRIORITY_WEIGHT_KEY not in all_keys:
            return infos
        reference = next(
            info.get(_PRIORITY_WEIGHT_KEY)
            for info, member_keys in zip(infos, keys)
            if _PRIORITY_WEIGHT_KEY in member_keys
        )
        filled = []
        for info, member_keys, sample in zip(infos, keys, index_parts):
            if _PRIORITY_WEIGHT_KEY not in member_keys:
                info = info.clone(False)
                info.set(
                    _PRIORITY_WEIGHT_KEY,
                    torch.ones(
                        sample.shape[:1],
                        dtype=reference.dtype,
                        device=reference.device,
                    ),
                )
            filled.append(info)
        return filled

    def can_sample(self, storage: StorageEnsemble, batch_size: int) -> bool:
        """Returns whether the selected ensemble strategy can serve a batch."""
        if self._fractions is not None:
            try:
                counts = self.quota_counts(batch_size)
            except ValueError:
                return False
            return all(
                sampler.can_sample(member_storage, count)
                for count, sampler, member_storage in zip(
                    counts, self._samplers, storage._storages
                )
                if count
            )
        if batch_size % self.num_buffer_sampled:
            return False
        sub_batch_size = batch_size // self.num_buffer_sampled
        readiness = [
            sampler.can_sample(member_storage, sub_batch_size)
            for member_storage, sampler in zip(storage._storages, self._samplers)
        ]
        if self.sample_from_all:
            return all(readiness)
        if isinstance(self.p, str) or self.p is None:
            return any(readiness)
        return any(
            ready and probability > 0
            for ready, probability in zip(readiness, self.p.tolist())
        )

    def dumps(self, path: Path):
        path = Path(path).absolute()
        path.mkdir(parents=True, exist_ok=True)
        for i, sampler in enumerate(self._samplers):
            sampler.dumps(path / str(i))
        torch.save(self._fractions, path / "fractions.pt")

    def loads(self, path: Path):
        path = Path(path).absolute()
        for i, sampler in enumerate(self._samplers):
            sampler.loads(path / str(i))
        fractions = path / "fractions.pt"
        if fractions.exists():
            self.fractions = torch.load(fractions, weights_only=True)

    def state_dict(self) -> dict[str, Any]:
        state_dict = OrderedDict()
        for i, sampler in enumerate(self._samplers):
            state_dict[str(i)] = sampler.state_dict()
        state_dict["fractions"] = (
            self._fractions.clone() if self._fractions is not None else None
        )
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for i, sampler in enumerate(self._samplers):
            sampler.load_state_dict(state_dict[str(i)])
        if "fractions" in state_dict:
            self.fractions = state_dict["fractions"]

    def _empty(self):
        for sampler in self._samplers:
            sampler._empty()

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
                    "A floating point index was received when an integer dtype was expected."
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
        if self._fractions is not None:
            fractions = (
                self._fractions[torch.as_tensor(index)]
                if isinstance(index, list)
                else self._fractions[index]
            )
            if not float(fractions.sum()):
                raise ValueError("The selected members all have a fraction of 0.")
            return SamplerEnsemble(*samplers, fractions=fractions / fractions.sum())
        p = self._p[index] if isinstance(self._p, torch.Tensor) else self._p
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
        return f"{self.__class__.__name__}(\n{samplers})"
