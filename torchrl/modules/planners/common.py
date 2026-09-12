# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from tensordict import NestedKey, TensorDictBase

from torchrl.modules import SafeModule

if TYPE_CHECKING:
    from torchrl.envs.common import EnvBase


def _mask_post_done_reward(
    tensordict: TensorDictBase,
    reward_key: NestedKey = ("next", "reward"),
    done_key: NestedKey = ("next", "done"),
    *,
    time_dim: int | None = None,
) -> torch.Tensor:
    """Zero rewards that follow the first ``done`` along the time dimension.

    The reward at the first ``done`` step is kept. ``done`` is used (not
    ``terminated``) so a truncated episode also stops contributing to the
    planning score.
    """
    reward = tensordict.get(reward_key)
    done = tensordict.get(done_key)
    if time_dim is None:
        names = tensordict.names
        if names is not None and "time" in names:
            time_dim = names.index("time")
        else:
            time_dim = -2
    if done.shape != reward.shape:
        done = done.expand_as(reward)
    done_int = done.to(dtype=torch.int64)
    already_done = done_int.cumsum(dim=time_dim) > done_int
    return torch.where(already_done, torch.zeros_like(reward), reward)


class MPCPlannerBase(SafeModule, metaclass=abc.ABCMeta):
    """MPCPlannerBase abstract Module.

    This class inherits from :obj:`SafeModule`. Provided a :obj:`TensorDict`, this module will perform a Model Predictive Control (MPC) planning step.
    At the end of the planning step, the :obj:`MPCPlanner` will return a proposed action.

    Imagined rollouts keep a full planning horizon even after a candidate
    hits ``done``. Rewards after the first :obj:`("next", "done")` are ignored
    when scoring those trajectories.

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
