# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.storages import StorageEnsemble

from .base import Sampler


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

    .. warning::
      The indices provided in the info dictionary are placed in a :class:`~tensordict.TensorDict` with
      keys ``index`` and ``buffer_ids`` that allow the upper :class:`~torchrl.data.ReplayBufferEnsemble`
      and :class:`~torchrl.data.StorageEnsemble` objects to retrieve the data.
      This format is different from with other samplers which usually return indices
      as regular tensors.

    """

    def __init__(
        self,
        *samplers,
        p: list[float] | torch.Tensor | Literal["sampleable"] | None = None,
        sample_from_all: bool = False,
        num_buffer_sampled: int | None = None,
    ):
        self._rng_private = None
        self._samplers = samplers
        self.sample_from_all = sample_from_all
        if sample_from_all and p is not None:
            raise RuntimeError(
                "Cannot pass both `p` argument and `sample_from_all=True`."
            )
        self.p = p
        self.num_buffer_sampled = num_buffer_sampled

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
    def num_buffer_sampled(self):
        value = self.__dict__.get("_num_buffer_sampled", None)
        if value is None:
            value = self.__dict__["_num_buffer_sampled"] = len(self._samplers)
        return value

    @num_buffer_sampled.setter
    def num_buffer_sampled(self, value):
        self.__dict__["_num_buffer_sampled"] = value

    def sample(self, storage, batch_size):
        if batch_size % self.num_buffer_sampled > 0:
            raise ValueError("The batch size must be divisible by num_buffer_sampled.")
        if not isinstance(storage, StorageEnsemble):
            raise TypeError("SamplerEnsemble requires a StorageEnsemble.")
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
                        float(
                            torch.as_tensor(
                                sampler._sampleable_count(
                                    member_storage, sub_batch_size
                                )
                            ).item()
                        )
                        for member_storage, sampler in zip(
                            storage._storages, self._samplers
                        )
                    ],
                    dtype=torch.float,
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
        infos = torch.stack(infos)
        return samples, infos

    def can_sample(self, storage: StorageEnsemble, batch_size: int) -> bool:
        """Returns whether the selected ensemble strategy can serve a batch."""
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
        for i, sampler in enumerate(self._samplers):
            sampler.dumps(path / str(i))

    def loads(self, path: Path):
        path = Path(path).absolute()
        for i, sampler in enumerate(self._samplers):
            sampler.loads(path / str(i))

    def state_dict(self) -> dict[str, Any]:
        state_dict = OrderedDict()
        for i, sampler in enumerate(self._samplers):
            state_dict[str(i)] = sampler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for i, sampler in enumerate(self._samplers):
            sampler.load_state_dict(state_dict[str(i)])

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
