# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tensordict import is_tensor_collection, TensorDict
from tensordict.utils import NestedKey
from torchrl.data.replay_buffers.storages import Storage

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


from .base import Sampler


class PromptGroupSampler(Sampler):
    """A sampler that draws complete groups of items sharing a common key.

    This sampler partitions the storage into groups whose items share the same
    value under ``group_key`` (for LLM post-training, the prompt or query). Every
    call to :meth:`~torchrl.data.ReplayBuffer.sample` returns
    ``samples_per_group`` items for each of ``num_groups`` selected groups, so a
    batch is laid out as balanced groups rather than independent items. This is
    the layout required by group-relative objectives such as
    :class:`~torchrl.objectives.llm.GRPOLoss`.

    Sampling never consumes the storage, so past generations for a group remain
    available and can be replayed across policy updates. Combined with a
    persistent replay buffer (one that is not emptied between iterations), this
    turns an on-policy GRPO loop into the replay-enhanced regime of RePO
    ("RePO: Replay-Enhanced Policy Optimization", Li et al. 2025,
    https://arxiv.org/abs/2506.09340), where each update mixes fresh on-policy
    groups with off-policy groups retrieved from the buffer.

    Keyword Args:
        num_groups (int, optional): the number of distinct groups to draw per
            batch. Exactly one of ``num_groups`` or ``samples_per_group`` must be
            provided; the other is inferred from the ``batch_size`` passed to
            :meth:`~torchrl.data.ReplayBuffer.sample`.
        samples_per_group (int, optional): the number of items to draw from each
            selected group. Exactly one of ``num_groups`` or
            ``samples_per_group`` must be provided.
        group_key (NestedKey, optional): the tensordict key identifying the group
            each item belongs to. Stored values may be integers (e.g. a prompt
            id) or strings (e.g. the prompt text). Defaults to ``"query"``.
        strategy (str, optional): the retrieval strategy. One of:

            - ``"random"`` (default): groups are chosen uniformly at random and
              items within a group are drawn uniformly at random.
            - ``"recency"``: the most recently inserted items are drawn from
              each group.
            - ``"reward"``: the highest-reward items are drawn from each group.
            - ``"variance"``: the fixed-size subset that maximizes reward
              variance is drawn from each group, breaking ties by total reward.
              This targets the vanishing-gradient case described by RePO.

        reward_key (NestedKey, optional): the key holding a numeric reward,
            required by the ``"reward"`` and ``"variance"`` strategies. It is
            reduced to one scalar per item by averaging over any trailing
            dimensions. Defaults to ``("next", "reward")``.
        cache_groups (bool, optional): if ``True`` (default), the group index is
            cached and rebuilt only when items are added to the storage. Set to
            ``False`` if the stored group values may change in place.

    .. note:: This sampler supports single-dimensional TensorDict-backed
        storages, including :class:`~torchrl.data.LazyTensorStorage`,
        :class:`~torchrl.data.LazyMemmapStorage`, and
        :class:`~torchrl.data.LazyStackStorage`. Plain
        :class:`~torchrl.data.ListStorage` is unsupported because its slices
        return Python lists.

    .. warning:: When a group holds fewer than ``samples_per_group`` items (or
        the storage holds fewer than ``num_groups`` groups), the missing draws
        are completed by sampling with replacement and a warning is raised once.

    .. seealso:: :class:`~torchrl.objectives.llm.MCAdvantage`, the group-relative
        advantage engine these batches feed into.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyStackStorage, ReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import PromptGroupSampler
        >>> rb = ReplayBuffer(
        ...     storage=LazyStackStorage(100),
        ...     sampler=PromptGroupSampler(num_groups=2, group_key="prompt"),
        ...     batch_size=8,
        ... )
        >>> data = TensorDict(
        ...     {
        ...         "prompt": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        ...         "reward": torch.arange(12.0),
        ...     },
        ...     batch_size=[12],
        ... )
        >>> _ = rb.extend(data)
        >>> sample = rb.sample()
        >>> int(sample["prompt"].unique().numel())
        2
        >>> int(sample.shape[0])
        8
    """

    def __init__(
        self,
        *,
        num_groups: int | None = None,
        samples_per_group: int | None = None,
        group_key: NestedKey = "query",
        strategy: Literal["random", "recency", "reward", "variance"] = "random",
        reward_key: NestedKey = ("next", "reward"),
        cache_groups: bool = True,
    ) -> None:
        if (num_groups is None) == (samples_per_group is None):
            raise TypeError(
                "Exactly one of num_groups or samples_per_group must be provided, "
                f"got num_groups={num_groups} and samples_per_group={samples_per_group}."
            )
        value_name = "num_groups" if num_groups is not None else "samples_per_group"
        value = num_groups if num_groups is not None else samples_per_group
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{value_name} must be a positive integer, got {value!r}.")
        if value <= 0:
            raise ValueError(f"{value_name} must be a positive integer, got {value!r}.")
        if strategy not in ("random", "recency", "reward", "variance"):
            raise ValueError(
                f"Unknown strategy={strategy!r}. Expected one of 'random', "
                "'recency', 'reward', 'variance'."
            )
        self.num_groups = int(num_groups) if num_groups is not None else None
        self.samples_per_group = (
            int(samples_per_group) if samples_per_group is not None else None
        )
        self.group_key = group_key
        self.strategy = strategy
        self.reward_key = reward_key
        self.cache_groups = cache_groups
        self._group_index: dict[Any, torch.Tensor] | None = None
        self._row_rewards: torch.Tensor | None = None
        self._index_len: int = 0
        self._cache_storage_id: int | None = None
        self._position_recency: torch.Tensor | None = None
        self._recency_clock: int = 0
        self._warned_small_group: bool = False

    @property
    def _needs_reward(self) -> bool:
        return self.strategy in ("reward", "variance")

    def _shape(self, batch_size: int) -> tuple[int, int]:
        if self.num_groups is not None:
            num_groups = self.num_groups
            if batch_size % num_groups != 0:
                raise ValueError(
                    f"batch_size={batch_size} is not divisible by "
                    f"num_groups={num_groups}."
                )
            return num_groups, batch_size // num_groups
        samples_per_group = self.samples_per_group
        if batch_size % samples_per_group != 0:
            raise ValueError(
                f"batch_size={batch_size} is not divisible by "
                f"samples_per_group={samples_per_group}."
            )
        return batch_size // samples_per_group, samples_per_group

    def _randperm(self, n: int) -> torch.Tensor:
        device = self._rng.device if self._rng is not None else None
        return torch.randperm(n, generator=self._rng, device=device).cpu()

    def _randint(self, n: int, k: int) -> torch.Tensor:
        device = self._rng.device if self._rng is not None else None
        return torch.randint(n, (k,), generator=self._rng, device=device).cpu()

    @staticmethod
    def _to_key_list(values: Any) -> list:
        if hasattr(values, "tolist"):
            values = values.tolist()
        else:
            values = list(values)
        return [tuple(value) if isinstance(value, list) else value for value in values]

    def _maybe_build_index(self, storage: Storage) -> None:
        length = len(storage)
        if (
            self.cache_groups
            and self._group_index is not None
            and length == self._index_len
            and self._cache_storage_id == id(storage)
        ):
            return
        data = storage[:]
        if not is_tensor_collection(data):
            raise TypeError(
                f"{type(self).__name__} requires a single-dimensional storage "
                "whose slices return a tensor collection, such as "
                "LazyTensorStorage or LazyStackStorage. Plain ListStorage is "
                "not supported."
            )
        keys = self._to_key_list(data.get(self.group_key))
        if len(keys) != length:
            raise RuntimeError(
                f"Expected group_key={self.group_key!r} to contain one value per "
                f"storage item, got {len(keys)} values for {length} items."
            )
        group_index: dict[Any, list[int]] = defaultdict(list)
        for position, key in enumerate(keys):
            group_index[key].append(position)
        self._group_index = {
            key: torch.tensor(positions, dtype=torch.long)
            for key, positions in group_index.items()
        }
        if self._needs_reward:
            reward = data.get(self.reward_key)
            if not isinstance(reward, torch.Tensor):
                raise TypeError(
                    f"The {self.strategy!r} strategy requires reward_key="
                    f"{self.reward_key} to hold a tensor, got {type(reward)}."
                )
            self._row_rewards = (
                reward.reshape(reward.shape[0], -1).float().mean(-1).cpu()
            )
        self._index_len = len(keys)
        self._cache_storage_id = id(storage)
        if self.strategy == "recency":
            self._ensure_recency(length)

    def _select_groups(self, num_groups: int) -> list:
        keys = list(self._group_index)
        n = len(keys)
        if n >= num_groups:
            return [keys[i] for i in self._randperm(n)[:num_groups].tolist()]
        self._warn_small()
        selected = [keys[i] for i in self._randperm(n).tolist()]
        selected.extend(keys[i] for i in self._randint(n, num_groups - n).tolist())
        return selected

    def _select_max_variance(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        rewards = self._row_rewards[positions].double()
        order = torch.argsort(rewards)
        values = rewards[order]
        n = values.numel()

        prefix = torch.cat([values.new_zeros(1), values.cumsum(0)])
        prefix_sq = torch.cat([values.new_zeros(1), values.square().cumsum(0)])
        num_low = torch.arange(k + 1)
        num_high = k - num_low
        totals = prefix[num_low] + prefix[n] - prefix[n - num_high]
        totals_sq = prefix_sq[num_low] + prefix_sq[n] - prefix_sq[n - num_high]
        variances = totals_sq / k - (totals / k).square()

        # A maximum-variance fixed-size subset consists of some of the lowest
        # rewards and some of the highest rewards. Among equal-variance splits,
        # RePO selects the one with the highest total reward.
        max_variance = variances.max()
        best_num_low = torch.where(
            torch.isclose(variances, max_variance, rtol=1e-12, atol=1e-12),
            totals,
            totals.new_full((), -torch.inf),
        ).argmax()
        rank = torch.arange(n)
        selected = (rank < best_num_low) | (rank >= n - (k - best_num_low))
        return positions[order[selected]]

    def _select_within(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        n = positions.numel()
        if n < k:
            self._warn_small()
            selected = self._select_within(positions, n)
            replacement = positions[self._randint(n, k - n)]
            return torch.cat([selected, replacement])
        if self.strategy == "recency":
            recency = self._position_recency[positions]
            return positions[torch.argsort(recency)[-k:]]
        if self.strategy == "reward":
            top = torch.topk(self._row_rewards[positions], k).indices
            return positions[top]
        if self.strategy == "variance":
            return self._select_max_variance(positions, k)
        return positions[self._randperm(n)[:k]]

    def _warn_small(self) -> None:
        if not self._warned_small_group:
            warnings.warn(
                "A group (or the set of groups) was smaller than the requested "
                "sample size; completing the draw with replacement. Add more data "
                "or lower num_groups/samples_per_group to avoid this.",
                stacklevel=2,
            )
            self._warned_small_group = True

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        if storage.ndim > 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports single-dimensional storages, "
                f"got storage.ndim={storage.ndim}."
            )
        num_groups, samples_per_group = self._shape(batch_size)
        self._maybe_build_index(storage)
        selected = self._select_groups(num_groups)
        index = torch.cat(
            [
                self._select_within(self._group_index[key], samples_per_group)
                for key in selected
            ],
            0,
        )
        return index, {}

    def _invalidate_cache(self) -> None:
        self._group_index = None
        self._row_rewards = None
        self._index_len = 0
        self._cache_storage_id = None

    def _record_update(self, index: int | torch.Tensor) -> None:
        if self.strategy != "recency":
            return
        index = torch.as_tensor(index, dtype=torch.long).reshape(-1).cpu()
        if not index.numel():
            return
        required_size = int(index.max()) + 1
        if self._position_recency is None:
            self._position_recency = torch.full((required_size,), -1, dtype=torch.long)
        elif self._position_recency.numel() < required_size:
            self._position_recency = torch.cat(
                [
                    self._position_recency,
                    torch.full(
                        (required_size - self._position_recency.numel(),),
                        -1,
                        dtype=torch.long,
                    ),
                ]
            )
        for position in index.tolist():
            self._recency_clock += 1
            self._position_recency[position] = self._recency_clock

    def _ensure_recency(self, length: int) -> None:
        if self._position_recency is None:
            self._position_recency = torch.arange(length, dtype=torch.long)
            self._recency_clock = max(self._recency_clock, length)
            return
        if self._position_recency.numel() < length:
            self._position_recency = torch.cat(
                [
                    self._position_recency,
                    torch.full(
                        (length - self._position_recency.numel(),),
                        -1,
                        dtype=torch.long,
                    ),
                ]
            )
        recency = self._position_recency[:length]
        missing = (recency < 0).nonzero(as_tuple=True)[0]
        if not missing.numel():
            return
        known = recency[recency >= 0]
        if known.numel():
            stop = int(known.min())
            start = stop - missing.numel()
            recency[missing] = torch.arange(start, stop, dtype=torch.long)
        else:
            recency[missing] = torch.arange(missing.numel(), dtype=torch.long)
            self._recency_clock = max(self._recency_clock, missing.numel())

    def extend(self, index: torch.Tensor) -> None:
        self._record_update(index)
        self._invalidate_cache()

    def add(self, index: int) -> None:
        self._record_update(index)
        self._invalidate_cache()

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        self._record_update(index)
        self._invalidate_cache()

    def __getstate__(self):
        state = super().__getstate__()
        state["_group_index"] = None
        state["_row_rewards"] = None
        state["_index_len"] = 0
        state["_cache_storage_id"] = None
        return state

    def _empty(self) -> None:
        self._invalidate_cache()
        self._position_recency = None
        self._recency_clock = 0
        self._warned_small_group = False

    def state_dict(self) -> dict[str, Any]:
        if self._position_recency is None:
            return {}
        return {
            "position_recency": self._position_recency.clone(),
            "recency_clock": self._recency_clock,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        position_recency = state_dict.get("position_recency")
        self._position_recency = (
            position_recency.clone().cpu() if position_recency is not None else None
        )
        self._recency_clock = int(state_dict.get("recency_clock", 0))
        self._invalidate_cache()

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        TensorDict(self.state_dict()).memmap(path)

    def loads(self, path):
        state_dict = TensorDict.load_memmap(path).to_dict()
        self.load_state_dict(state_dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_groups={self.num_groups}, "
            f"samples_per_group={self.samples_per_group}, "
            f"group_key={self.group_key}, strategy={self.strategy!r})"
        )
