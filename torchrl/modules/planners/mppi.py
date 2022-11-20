# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict.tensordict import TensorDictBase, TensorDict

from torchrl.envs import EnvBase
from torchrl.modules.planners.common import MPCPlannerBase


class MPPIlanner(MPCPlannerBase):
    def __init__(
        self,
        env: EnvBase,
        planning_horizon: int,
        optim_steps: int,
        num_candidates: int,
        top_k: int,
        reward_key: str = "reward",
        action_key: str = "action",
    ):
        super().__init__(env=env, action_key=action_key)
        self.planning_horizon = planning_horizon
        self.optim_steps = optim_steps
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.reward_key = reward_key

    def planning(self, tensordict: TensorDictBase) -> torch.Tensor:
        batch_size = tensordict.batch_size
        action_shape = (
            *batch_size,
            self.num_candidates,
            self.planning_horizon,
            *self.action_spec.shape,
        )
        action_stats_shape = (
            *batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
        )
        action_topk_shape = (
            *batch_size,
            self.top_k,
            self.planning_horizon,
            *self.action_spec.shape,
        )
        TIME_DIM = len(self.action_spec.shape) - 3
        K_DIM = len(self.action_spec.shape) - 4
        expanded_original_tensordict = (
            tensordict.unsqueeze(-1)
            .expand(*batch_size, self.num_candidates)
            .to_tensordict()
        )
        _action_means = torch.zeros(
            *action_stats_shape,
            device=tensordict.device,
            dtype=self.env.action_spec.dtype,
        )
        _action_stds = torch.ones_like(_action_means)
        container = TensorDict(
            {
                "tensordict": expanded_original_tensordict,
                "stats": TensorDict(
                    {
                        "_action_means": _action_means,
                        "_action_stds": _action_stds,
                    },
                    [*batch_size, 1, self.planning_horizon],
                ),
            },
            batch_size,
        )

        for _ in range(self.optim_steps):
            actions_means = container.get(("stats", "_action_means"))
            actions_stds = container.get(("stats", "_action_stds"))
            actions = actions_means + actions_stds * torch.randn(
                *action_shape,
                device=actions_means.device,
                dtype=actions_means.dtype,
            )
            actions = self.env.action_spec.project(actions)
            optim_tensordict = container.get("tensordict").clone()
            policy = _PrecomputedActionsSequentialSetter(actions)
            optim_tensordict = self.env.rollout(
                max_steps=self.planning_horizon,
                policy=policy,
                auto_reset=False,
                tensordict=optim_tensordict,
            )

            sum_rewards = optim_tensordict.get(self.reward_key).sum(
                dim=TIME_DIM, keepdim=True
            )
            _, top_k = sum_rewards.topk(self.top_k, dim=K_DIM)
            top_k = top_k.expand(action_topk_shape)
            # get omega weights
            Omegas = (self.temp * vals).exp()

            best_actions = actions.gather(K_DIM, top_k)
            _action_means = (Omegas * best_actions).sum(dim=K_DIM, keepdim=True) / Omegas.sum(K_DIM, True)
            _action_stds = ((Omegas * (best_actions - _action_means).pow(2)).sum(dim=K_DIM, keepdim=True) / Omegas.sum(K_DIM, True)).sqrt()
            container.set_(
                ("stats", "_action_means"), _action_means
            )
            container.set_(
                ("stats", "_action_stds"), _action_stds
            )
        action_means = container.get(("stats", "_action_means"))
        return action_means[..., 0, 0, :]


class _PrecomputedActionsSequentialSetter:
    def __init__(self, actions):
        self.actions = actions
        self.cmpt = 0

    def __call__(self, tensordict):
        # checks that the step count is lower or equal to the horizon
        if self.cmpt >= self.actions.shape[-2]:
            raise ValueError("Precomputed actions sequence is too short")
        tensordict = tensordict.set("action", self.actions[..., self.cmpt, :])
        self.cmpt += 1
        return tensordict
