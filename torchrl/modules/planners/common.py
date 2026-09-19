# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import NestedKey, TensorDictBase, unravel_key

from torchrl._utils import _ends_with
from torchrl.modules import SafeModule

if TYPE_CHECKING:
    from torchrl.envs.common import EnvBase


def _normalize_done_keys(
    done_key: NestedKey | Sequence[NestedKey],
) -> tuple[NestedKey, ...]:
    if isinstance(done_key, str):
        return (done_key,)
    if isinstance(done_key, tuple) and (not done_key or isinstance(done_key[0], str)):
        return (unravel_key(done_key),)
    return tuple(unravel_key(key) for key in done_key)


def _done_group_prefix(key: NestedKey) -> tuple[str, ...]:
    if isinstance(key, str):
        return ()
    return key[:-1]


def _effective_planning_done_keys(
    done_key: NestedKey | Sequence[NestedKey],
) -> tuple[NestedKey, ...]:
    keys = _normalize_done_keys(done_key)
    prefixes = tuple(_done_group_prefix(key) for key in keys)
    # EnvBase applies done-group precedence recursively.
    return tuple(
        key
        for key, prefix in zip(keys, prefixes)
        if not any(
            len(other) < len(prefix) and prefix[: len(other)] == other
            for other in prefixes
        )
    )


def _planning_done_keys(env: EnvBase) -> list[NestedKey]:
    """Return rollout ``done`` keys used to mask post-done planning rewards.

    Only keys that end with ``"done"`` are used, so a truncated episode
    (``done`` and not ``terminated``) still stops contributing. Keys are
    taken from the environment and prefixed with ``"next"``; a root
    ``"done"`` is not invented when the env only exposes a nested group.

    An ancestor ``done`` group takes precedence over its descendants
    (``("next", "done")`` over ``("next", "agent", "done")``, or
    ``("next", "team", "done")`` over ``("next", "team", "agent", "done")``).
    Independent sibling groups are combined with a logical or. This
    matches :func:`~torchrl.envs.utils._terminated_or_truncated`.
    """
    keys = [
        unravel_key(("next", key)) for key in env.done_keys if _ends_with(key, "done")
    ]
    return list(_effective_planning_done_keys(keys))


def _mask_post_done_reward(
    tensordict: TensorDictBase,
    reward_key: NestedKey = ("next", "reward"),
    done_key: NestedKey | Sequence[NestedKey] = ("next", "done"),
    *,
    time_dim: int | None = None,
) -> torch.Tensor:
    """Zero rewards that follow the first ``done`` along the time dimension.

    The reward at the first ``done`` step is kept. ``done`` is used (not
    ``terminated``) so a truncated episode also stops contributing to the
    planning score. An ancestor ``done`` group takes precedence over its
    descendants; otherwise several ``done`` keys are combined with a
    logical or.
    """
    reward = tensordict.get(reward_key)
    done = None
    for key in _effective_planning_done_keys(done_key):
        flag = tensordict.get(key)
        if flag is None:
            raise KeyError(f"Done key {key!r} not found in tensordict.")
        if flag.shape != reward.shape:
            flag = flag.expand_as(reward)
        done = flag if done is None else torch.logical_or(done, flag)
    if done is None:
        return reward
    if time_dim is None:
        names = tensordict.names
        if names is not None and "time" in names:
            time_dim = names.index("time")
        else:
            time_dim = -2
    done_int = done.to(dtype=torch.int64)
    already_done = done_int.cumsum(dim=time_dim) > done_int
    return torch.where(already_done, torch.zeros_like(reward), reward)


class MPCPlannerBase(SafeModule, metaclass=abc.ABCMeta):
    """MPCPlannerBase abstract Module.

    This class inherits from :obj:`SafeModule`. Provided a :obj:`TensorDict`, this module will perform a Model Predictive Control (MPC) planning step.
    At the end of the planning step, the :obj:`MPCPlanner` will return a proposed action.

    Imagined rollouts keep a full planning horizon even after a candidate
    hits ``done``. Rewards after the first environment ``done`` flag
    (termination or truncation) are ignored when scoring those
    trajectories. An ancestor ``done`` group takes precedence over its
    descendants, so a finished agent does not mask a still-running
    parent return. Nested-only environments use the nested ``done``
    key(s).

    Args:
        env (EnvBase): The environment to perform the planning step on (Can be :obj:`ModelBasedEnvBase` or :obj:`EnvBase`).
        action_key (NestedKey, optional): The key that will point to the computed action.
    """

    def __init__(
        self,
        env: EnvBase,
        action_key: NestedKey = "action",
    ):
        # Check if env is stateless
        if env.batch_locked:
            raise ValueError(
                "Environment is batch_locked. MPCPlanners need an environment that accepts batched inputs with any batch size"
            )
        out_keys = [action_key]
        in_keys = list(env.observation_spec.keys(True, True))
        super().__init__(env, in_keys=in_keys, out_keys=out_keys)
        self.env = env
        self.action_spec = env.action_spec
        self.to(env.device)

    @abc.abstractmethod
    def planning(self, td: TensorDictBase) -> torch.Tensor:
        """Performs the MPC planning step.

        Args:
            td (TensorDict): The TensorDict to perform the planning step on.
        """
        raise NotImplementedError()

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if "params" in kwargs or "vmap" in kwargs:
            raise ValueError(
                "MPCPlannerBase does not currently support functional programming."
            )
        action = self.planning(tensordict)
        action = self.action_spec.project(action)
        tensordict_out = self._write_to_tensordict(
            tensordict,
            (action,),
            tensordict_out,
        )
        return tensordict_out
