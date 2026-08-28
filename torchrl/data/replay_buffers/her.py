# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Hindsight Experience Replay (HER) replay buffer.

Reference: Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017.
https://arxiv.org/abs/1707.01495
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
from tensordict.base import TensorDictBase
from tensordict.utils import NestedKey

from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler, Sampler
from torchrl.data.replay_buffers.storages import Storage

try:
    from enum import StrEnum  # Python >= 3.11
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: D101 - backport of enum.StrEnum
        pass


class HindsightStrategy(StrEnum):
    """Goal selection strategy for :class:`HERReplayBuffer`.

    Attributes:
        FUTURE: sample a goal from a future transition in the same episode
            (recommended; strongest empirically).
        FINAL: use the final achieved state of the episode as the goal.
        EPISODE: sample any achieved state from the same episode uniformly.
        RANDOM: sample a random achieved state from the entire buffer.
    """

    FUTURE = "future"
    FINAL = "final"
    EPISODE = "episode"
    RANDOM = "random"


class HERReplayBuffer(TensorDictReplayBuffer):
    """Hindsight Experience Replay (HER) replay buffer.

    Applies goal relabeling at sample time for goal-conditioned RL. For a
    fraction ``her_ratio`` of each sampled batch, the desired goal is
    replaced with an achieved goal drawn from the same episode (or from the
    full buffer when ``strategy="random"``), and the reward is recomputed
    via ``reward_fn``. The remaining ``1 - her_ratio`` fraction is returned
    with the original goals and rewards.

    Episode boundaries are detected from ``end_key`` (done flag) and the
    corresponding ``terminated`` flag in storage. At least one must be
    present; if neither is found a :class:`KeyError` is raised at sample time.
    See :ref:`Environment-API` for the semantics of ``"done"`` /
    ``"terminated"`` / ``"truncated"``.

    .. note::
        Episode-boundary detection currently assumes a 1D storage. If the
        buffer is full and a trajectory spans the wrap-around point, the
        write cursor is treated as a synthetic episode boundary, so goals
        will not be sampled across the wrap. ``nonzero()`` on the done
        signal is a CPU/GPU sync that breaks ``torch.compile`` graphs;
        there is no efficient way around this for sparse boundary detection.

    Args:
        reward_fn (Callable[[TensorDictBase], Tensor]): receives a tensordict
            containing the *relabeled* transition (with the new
            ``goal_key`` already set) and must return a reward tensor with
            shape ``(*batch, 1)`` or ``(*batch,)``.

    Keyword Args:
        her_ratio (float): fraction of sampled transitions to relabel. Must
            be in ``[0, 1]``. Default: ``0.8``.
        strategy (HindsightStrategy or str): one of ``"future"``,
            ``"final"``, ``"episode"``, ``"random"``. Default: ``"future"``.
        goal_key (NestedKey): key for the desired goal. Default:
            ``"desired_goal"``.
        achieved_goal_key (NestedKey): key for the achieved goal. Default:
            ``"achieved_goal"``.
        reward_key (NestedKey): key where the reward is stored. Defaults to
            ``("next", "reward")`` which is the TorchRL convention.
        end_key (NestedKey): key indicating episode boundaries (done flag).
            The corresponding ``terminated`` key is derived by replacing the
            last component with ``"terminated"``. Default: ``("next", "done")``.
        sampler (Sampler, optional): index sampler used to draw transitions
            (any standard :class:`~torchrl.data.Sampler` works). The HER
            relabeling logic is applied on top of the indices returned by
            this sampler. Defaults to :class:`~torchrl.data.RandomSampler`.
        **kwargs: forwarded to :class:`~torchrl.data.TensorDictReplayBuffer`.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import HERReplayBuffer, LazyMemmapStorage
        >>>
        >>> def reward_fn(td):
        ...     dist = (td["achieved_goal"] - td["desired_goal"]).norm(dim=-1, keepdim=True)
        ...     return (dist < 0.05).float()
        >>>
        >>> rb = HERReplayBuffer(
        ...     reward_fn=reward_fn,
        ...     storage=LazyMemmapStorage(10_000),
        ...     batch_size=256,
        ... )
        >>> # Store several transitions, marking the last one as done
        >>> for i in range(4):
        ...     td = TensorDict(
        ...         {
        ...             "observation": torch.randn(4),
        ...             "desired_goal": torch.zeros(3),
        ...             "achieved_goal": torch.randn(3),
        ...             "action": torch.randn(2),
        ...             "next": {
        ...                 "observation": torch.randn(4),
        ...                 "desired_goal": torch.zeros(3),
        ...                 "achieved_goal": torch.randn(3),
        ...                 "reward": torch.zeros(1),
        ...                 "done": torch.tensor([i == 3]),
        ...             },
        ...         },
        ...         batch_size=[],
        ...     )
        ...     _ = rb.add(td)
        >>> batch = rb.sample(4)
        >>> batch["desired_goal"].shape
        torch.Size([4, 3])
        >>> batch[("next", "reward")].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        reward_fn: Callable[[TensorDictBase], torch.Tensor],
        *,
        her_ratio: float = 0.8,
        strategy: HindsightStrategy
        | Literal["future", "final", "episode", "random"] = "future",
        goal_key: NestedKey = "desired_goal",
        achieved_goal_key: NestedKey = "achieved_goal",
        reward_key: NestedKey = ("next", "reward"),
        end_key: NestedKey = ("next", "done"),
        sampler: Sampler | None = None,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= her_ratio <= 1.0:
            raise ValueError(f"her_ratio must be in [0, 1], got {her_ratio}")
        if sampler is None:
            sampler = RandomSampler()
        super().__init__(sampler=sampler, **kwargs)
        self.reward_fn = reward_fn
        self.her_ratio = her_ratio
        self.strategy = HindsightStrategy(strategy)
        self.end_key = end_key
        self.goal_key = goal_key
        self.achieved_goal_key = achieved_goal_key
        self.reward_key = reward_key
        self._keys_validated: bool = False
        self._episode_ends_cache: torch.Tensor | None = None
        self._last_cache_key: tuple = (-1, None, False)

    # -- key helpers ---------------------------------------------------------

    def _terminated_key(self) -> NestedKey:
        if isinstance(self.end_key, tuple):
            return self.end_key[:-1] + ("terminated",)
        return "terminated"

    @staticmethod
    def _next_key(key: NestedKey) -> tuple:
        if isinstance(key, tuple):
            return ("next",) + key
        return ("next", key)

    # -- episode cache -------------------------------------------------------

    def _storage_cache_key(self, storage: Storage) -> tuple:
        return (
            len(storage),
            getattr(storage, "_last_cursor", None),
            bool(getattr(storage, "_is_full", False)),
        )

    def _get_episode_ends(self, storage: Storage) -> torch.Tensor:
        key = self._storage_cache_key(storage)
        if key != self._last_cache_key:
            self._episode_ends_cache = self._build_episode_cache(storage)
            self._last_cache_key = key
        return self._episode_ends_cache

    def _build_episode_cache(self, storage: Storage) -> torch.Tensor:
        n = len(storage)
        if n == 0:
            return torch.zeros(0, dtype=torch.long)

        # HER assumes a 1D storage; multi-dim storages would require flattening
        # logic that is not currently in scope.
        if getattr(storage, "ndim", 1) != 1:
            raise NotImplementedError(
                "HERReplayBuffer currently only supports 1D storages "
                f"(got ndim={storage.ndim})."
            )

        # Fetch only the done/terminated fields rather than the whole storage.
        # For a TensorDict-backed storage, ``storage[:][key]`` materialises a
        # single field, not the full transition data.
        try:
            view = storage[:]
        except Exception as e:
            raise RuntimeError(
                "HERReplayBuffer could not read from storage to compute "
                "episode boundaries."
            ) from e

        done = view.get(self.end_key, None) if hasattr(view, "get") else None
        terminated = (
            view.get(self._terminated_key(), None) if hasattr(view, "get") else None
        )

        if done is None and terminated is None:
            raise KeyError(
                f"Neither {self.end_key!r} nor {self._terminated_key()!r} were "
                "found in storage. Ensure episode boundaries are stored under "
                "one of these keys. See "
                "https://pytorch.org/rl/main/reference/envs.html#environment-api "
                "for the semantics of done / terminated / truncated."
            )

        if done is not None and terminated is not None:
            boundary = done.bool() | terminated.bool()
        elif done is not None:
            boundary = done.bool()
        else:
            boundary = terminated.bool()

        # Move bookkeeping to CPU: sample indices live on CPU and the boundary
        # tensor is a single binary signal so the transfer is cheap even for
        # CUDA storages. ``nonzero`` is a sync op and intentionally breaks the
        # compile graph here; there isn't a sparse-boundary alternative.
        boundary = boundary.to(device="cpu").reshape(-1)[:n]

        episode_ends = boundary.nonzero(as_tuple=True)[0]

        is_full = bool(getattr(storage, "_is_full", False))
        if is_full:
            # When the storage has wrapped around, the write cursor delimits
            # the last in-progress trajectory. Treat it as a synthetic
            # boundary so goal sampling does not cross the wrap.
            cursor = getattr(storage, "_last_cursor", None)
            cursor_idx = _resolve_cursor(cursor)
            if cursor_idx is not None:
                cursor_idx = cursor_idx % n
                if cursor_idx not in episode_ends.tolist():
                    episode_ends = torch.cat(
                        [
                            episode_ends,
                            torch.tensor(
                                [cursor_idx],
                                dtype=episode_ends.dtype,
                                device=episode_ends.device,
                            ),
                        ]
                    )
                    episode_ends, _ = torch.sort(episode_ends)
        else:
            # Bound the last (possibly partial) episode at n-1.
            if len(episode_ends) == 0 or int(episode_ends[-1]) != n - 1:
                episode_ends = torch.cat(
                    [
                        episode_ends,
                        torch.tensor(
                            [n - 1],
                            dtype=episode_ends.dtype,
                            device=episode_ends.device,
                        ),
                    ]
                )
        return episode_ends

    # -- episode lookup ------------------------------------------------------

    def _get_episode_range(
        self,
        idx: torch.Tensor,
        episode_ends: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both idx and episode_ends live on CPU (see _build_episode_cache).
        idx_cpu = idx.to(episode_ends.device)
        ep_end_pos = torch.searchsorted(episode_ends, idx_cpu, right=False).clamp(
            max=len(episode_ends) - 1
        )
        ep_end = episode_ends[ep_end_pos]
        prev_pos = (ep_end_pos - 1).clamp(min=0)
        ep_start = torch.where(
            ep_end_pos == 0,
            torch.zeros_like(ep_end),
            episode_ends[prev_pos] + 1,
        )
        return ep_start, ep_end

    def _sample_goal_indices(
        self,
        her_idx: torch.Tensor,
        ep_starts: torch.Tensor,
        ep_ends: torch.Tensor,
        storage_len: int,
    ) -> torch.Tensor:
        n = len(her_idx)
        device = her_idx.device
        strategy = self.strategy

        if strategy == HindsightStrategy.FUTURE:
            span = (ep_ends - her_idx).clamp(min=0).float()
            offsets = (
                (torch.rand(n, device=device) * (span + 1))
                .long()
                .clamp(max=span.long())
            )
            return her_idx + offsets

        if strategy == HindsightStrategy.FINAL:
            return ep_ends.clone()

        if strategy == HindsightStrategy.EPISODE:
            span = (ep_ends - ep_starts + 1).float()
            offsets = (torch.rand(n, device=device) * span).long()
            return ep_starts + offsets

        if strategy == HindsightStrategy.RANDOM:
            return torch.randint(storage_len, (n,), device=device)

        raise ValueError(f"Unknown strategy: {strategy!r}")

    # -- validation ----------------------------------------------------------

    def _validate_storage_keys(self) -> None:
        if self._keys_validated or len(self._storage) == 0:
            return
        sample = self._storage.get(torch.tensor([0]))
        missing = []
        for key, role in (
            (self.goal_key, "goal_key"),
            (self.achieved_goal_key, "achieved_goal_key"),
        ):
            if sample.get(key, None) is None:
                missing.append(f"  {role}={key!r}")
        if missing:
            raise KeyError(
                "The following keys are not present in storage:\n"
                + "\n".join(missing)
                + "\nEnsure they are stored in every transition before sampling."
            )
        self._keys_validated = True

    # -- sampling ------------------------------------------------------------

    def _sample(self, batch_size: int) -> tuple[Any, dict]:
        self._validate_storage_keys()
        data, info = super()._sample(batch_size)

        episode_ends = self._get_episode_ends(self._storage)
        if len(episode_ends) == 0:
            return data, info

        idx = info["index"]
        if isinstance(idx, tuple):
            idx = idx[0]
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx)

        n = idx.shape[0]
        n_her = int(n * self.her_ratio)
        if n_her == 0:
            return data, info

        storage_idx = idx[:n_her]
        ep_starts, ep_ends = self._get_episode_range(storage_idx, episode_ends)
        goal_src_idx = self._sample_goal_indices(
            storage_idx, ep_starts, ep_ends, len(self._storage)
        )

        goal_src_tds = self._storage.get(goal_src_idx)
        achieved_goals = goal_src_tds.get(self.achieved_goal_key)

        her_slice = data[:n_her]
        with data.unlock_():
            her_slice.set_(self.goal_key, achieved_goals)
            next_goal_key = self._next_key(self.goal_key)
            if her_slice.get(next_goal_key, None) is not None:
                her_slice.set_(next_goal_key, achieved_goals)
            new_rewards = self.reward_fn(her_slice)
            her_slice.set_(self.reward_key, new_rewards)

        return data, info

    # -- empty / checkpoint --------------------------------------------------

    def empty(self, empty_write_count: bool = True) -> None:
        super().empty(empty_write_count=empty_write_count)
        self._episode_ends_cache = None
        self._last_cache_key = (-1, None, False)

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["_her"] = {
            "goal_key": self.goal_key,
            "achieved_goal_key": self.achieved_goal_key,
            "reward_key": self.reward_key,
            "her_ratio": self.her_ratio,
            "strategy": self.strategy.value,
            "episode_ends_cache": self._episode_ends_cache,
            "last_cache_key": self._last_cache_key,
        }
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        her_state = state_dict.pop("_her", None)
        super().load_state_dict(state_dict)
        if her_state is not None:
            self._episode_ends_cache = her_state.get("episode_ends_cache")
            self._last_cache_key = her_state.get("last_cache_key", (-1, None, False))

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"strategy={self.strategy.value!r}, "
            f"her_ratio={self.her_ratio}, "
            f"goal_key={self.goal_key!r}, "
            f"achieved_goal_key={self.achieved_goal_key!r}, "
            f"reward_key={self.reward_key!r}, "
            f"storage={self._storage})"
        )


def _resolve_cursor(cursor: Any) -> int | None:
    """Coerce a storage's ``_last_cursor`` attribute into a single int."""
    if cursor is None:
        return None
    if isinstance(cursor, torch.Tensor):
        if cursor.numel() == 0:
            return None
        return int(cursor.flatten()[-1].item())
    if isinstance(cursor, range):
        if len(cursor) == 0:
            return None
        return int(cursor[-1])
    if isinstance(cursor, (list, tuple)):
        if not cursor:
            return None
        return int(cursor[-1])
    try:
        return int(cursor)
    except (TypeError, ValueError):
        return None
