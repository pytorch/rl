# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.utils import NestedKey
from torchrl.data.replay_buffers.storages import Storage, TensorStorage

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


from .random import _default_staleness_weight
from .base import Sampler


class StalenessAwareSampler(Sampler):
    """A sampler that weights entries by freshness and filters stale entries.

    This sampler is designed for asynchronous training setups (e.g., async PPO)
    where collected data may come from older policy versions. It reads a
    ``policy_version`` field from the storage and uses it to:

    - **Hard gate**: Exclude entries whose staleness exceeds ``max_staleness``.
    - **Freshness weighting**: Sample proportionally to a weight that decays
      with staleness (default: ``1 / (staleness + 1)``).

    The training loop is responsible for updating :attr:`consumer_version`
    (typically after each optimizer step or weight update) so the sampler
    can compute staleness = ``consumer_version - policy_version``.

    Args:
        max_staleness (int, optional): Hard cutoff. Entries with
            ``staleness > max_staleness`` are excluded from sampling.
            ``-1`` (default) means no cutoff.
        staleness_weight_fn (callable, optional): A callable that maps a
            staleness tensor (int) to a weight tensor (float). Defaults to
            ``lambda s: 1.0 / (s.float() + 1.0)``.
        version_key (NestedKey, optional): The key in the storage holding
            the policy version. Defaults to ``"policy_version"``.

    Examples:
        >>> from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
        >>> from torchrl.data.replay_buffers.samplers import StalenessAwareSampler
        >>> sampler = StalenessAwareSampler(max_staleness=5)
        >>> buffer = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(1000),
        ...     sampler=sampler,
        ...     batch_size=32,
        ... )
        >>> # In training loop:
        >>> # sampler.consumer_version = current_training_step

    Integration with :class:`~torchrl.collectors.Collector` and
    :class:`~torchrl.envs.transforms.PolicyVersion`::

        from torchrl.collectors import Collector
        from torchrl.envs.transforms import PolicyVersion
        from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

        sampler = StalenessAwareSampler(max_staleness=10)
        buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(10_000),
            sampler=sampler,
            batch_size=256,
        )
        collector = Collector(
            env,
            policy,
            frames_per_batch=1000,
            total_frames=100_000,
            env_transforms=[PolicyVersion(collector)],
        )
        for step, data in enumerate(collector):
            buffer.extend(data)
            sampler.consumer_version = step
            batch = buffer.sample()
            # ... train on batch ...

    .. note::
        ``StalenessAwareSampler`` intentionally does **not** inherit from
        :class:`PrioritizedSampler`.  ``PrioritizedSampler`` maintains a
        segment-tree over per-transition TD-error priorities that are
        updated after each training step.  Staleness weighting is
        fundamentally different: weights are derived from a single scalar
        (``consumer_version``) and per-entry ``policy_version`` stamps,
        and are recomputed on every :meth:`sample` call rather than
        maintained incrementally.  Sharing the segment-tree machinery
        would add complexity without benefit.
    """

    def __init__(
        self,
        max_staleness: int = -1,
        staleness_weight_fn: callable | None = None,
        version_key: NestedKey = "policy_version",
    ):
        self._consumer_version = 0
        self._max_staleness = max_staleness
        self._version_key = version_key
        if staleness_weight_fn is None:
            staleness_weight_fn = _default_staleness_weight
        self._weight_fn = staleness_weight_fn

    @property
    def consumer_version(self) -> int:
        """The current training iteration / consumer version."""
        return self._consumer_version

    @consumer_version.setter
    def consumer_version(self, v: int):
        self._consumer_version = v

    def increment_consumer_version(self):
        """Increment the consumer version by 1."""
        self._consumer_version += 1

    @property
    def max_staleness(self) -> int:
        """The maximum allowed staleness. -1 means no limit."""
        return self._max_staleness

    @max_staleness.setter
    def max_staleness(self, v: int):
        self._max_staleness = v

    def _get_versions(self, storage: Storage, n: int):
        if isinstance(storage, TensorStorage):
            backing_storage = getattr(storage, "_storage", None)
            if backing_storage is not None:
                versions = backing_storage.get(self._version_key, None)
                if versions is not None:
                    if storage.ndim > 1:
                        versions = versions[: storage._len_along_dim0].flatten(
                            0, storage.ndim - 1
                        )
                    elif n != storage.max_size:
                        versions = versions[:n]
                    return versions

        # Fallback: storage[:n] materialises all data. This path is used for
        # non-TensorStorage backends (e.g. ListStorage). For hot-path
        # performance, prefer TensorStorage where versions are read directly.
        all_data = storage[:n]
        return all_data.get(self._version_key, None)

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        n = len(storage)
        if n == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)

        versions = self._get_versions(storage, n)
        if versions is None:
            raise KeyError(
                f"Could not find key {self._version_key!r} in the replay buffer "
                f"storage. Make sure data written to the buffer contains this key "
                "(e.g., by adding the PolicyVersion transform to the environment "
                "or enabling policy-version tracking on the collector)."
            )
        versions = versions.view(-1).float()

        # Compute staleness
        staleness = self._consumer_version - versions

        # Hard gate
        if self._max_staleness >= 0:
            valid_mask = staleness <= self._max_staleness
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                raise RuntimeError(
                    f"All {n} entries in the buffer exceed max_staleness="
                    f"{self._max_staleness} (consumer_version="
                    f"{self._consumer_version}). Increase max_staleness or "
                    f"update collector policy weights more frequently."
                )
        else:
            valid_indices = torch.arange(n, device=versions.device)

        # Compute freshness weights for valid entries
        valid_staleness = staleness[valid_indices]
        weights = self._weight_fn(valid_staleness)
        weights = weights / weights.sum()

        # Weighted sampling
        sampled_local = torch.multinomial(
            weights, batch_size, replacement=batch_size > len(valid_indices)
        )
        index = valid_indices[sampled_local]

        return index, {
            "staleness": staleness[index],
        }

    def _empty(self):
        self._consumer_version = 0

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        TensorDict(self.state_dict()).memmap(path)

    def loads(self, path):
        sd = TensorDict.load_memmap(path).to_dict()
        self.load_state_dict(sd)

    def state_dict(self) -> dict[str, Any]:
        return OrderedDict(
            consumer_version=self._consumer_version,
            max_staleness=self._max_staleness,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._consumer_version = int(state_dict["consumer_version"])
        self._max_staleness = int(state_dict["max_staleness"])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"consumer_version={self._consumer_version}, "
            f"max_staleness={self._max_staleness})"
        )
