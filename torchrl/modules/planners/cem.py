# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.envs import EnvBase
from torchrl.modules.planners import MPCPlannerBase

__all__ = ["CEMPlanner"]


class CEMPlanner(MPCPlannerBase):
    """CEMPlanner Module.

    Reference: The cross-entropy method for optimization, Botev et al. 2013

    This module will perform a CEM planning step when given a TensorDict containing initial states.
    The CEM planning step is performed by sampling actions from a Gaussian distribution with zero mean and unit variance.
    The sampled actions are then used to perform a rollout in the environment. The cumulative rewards obtained with the rollout is then
    ranked. We select the top-k episodes and use their actions to update the mean and standard deviation of the actions distribution.
    The CEM planning step is repeated for a specified number of steps.

    A call to the module returns the actions that empirically maximised the returns given a planning horizon

    Args:
        env (EnvBase): The environment to perform the planning step on (can be `ModelBasedEnv` or `EnvBase`).
        planning_horizon (int): The length of the simulated trajectories
        optim_steps (int): The number of optimization steps used by the MPC planner
        num_candidates (int): The number of candidates to sample from the Gaussian distributions.
        num_top_k_candidates (int): The number of top candidates to use to update the mean and standard deviation of the Gaussian distribution.
        reward_key (str, optional): The key in the TensorDict to use to retrieve the reward.
        action_key (str, optional): The key in the TensorDict to use to store the action.
    """

    def __init__(
        self,
        env: EnvBase,
        planning_horizon: int,
        optim_steps: int,
        num_candidates: int,
        num_top_k_candidates: int,
        reward_key: str = "reward",
        action_key: str = "action",
    ):
        super().__init__(env=env, action_key=action_key)
        self.planning_horizon = planning_horizon
        self.optim_steps = optim_steps
        self.num_candidates = num_candidates
        self.num_top_k_candidates = num_top_k_candidates
        self.reward_key = reward_key

    def planning(self, tensordict: TensorDictBase) -> torch.Tensor:
        batch_size = tensordict.batch_size
        expanded_original_tensordict = (
            tensordict.unsqueeze(-1)
            .expand(*batch_size, self.num_candidates)
            .reshape(-1)
        )
        flatten_batch_size = batch_size.numel()
        actions_means = torch.zeros(
            flatten_batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
            device=tensordict.device_safe(),
            dtype=self.env.action_spec.dtype,
        )
        actions_stds = torch.ones(
            flatten_batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
            device=tensordict.device_safe(),
            dtype=self.env.action_spec.dtype,
        )

        for _ in range(self.optim_steps):
            actions = actions_means + actions_stds * torch.randn(
                flatten_batch_size,
                self.num_candidates,
                self.planning_horizon,
                *self.action_spec.shape,
                device=tensordict.device_safe(),
                dtype=self.env.action_spec.dtype,
            )
            actions = actions.flatten(0, 1)
            actions = self.env.action_spec.project(actions)
            optim_tensordict = expanded_original_tensordict.to_tensordict()
            policy = PrecomputedActionsSequentialSetter(actions)
            optim_tensordict = self.env.rollout(
                max_steps=self.planning_horizon,
                policy=policy,
                auto_reset=False,
                tensordict=optim_tensordict,
            )
            rewards = (
                optim_tensordict.get(self.reward_key)
                .sum(dim=1)
                .reshape(flatten_batch_size, self.num_candidates)
            )
            _, top_k = rewards.topk(self.num_top_k_candidates, dim=1)

            best_actions = actions.unflatten(
                0, (flatten_batch_size, self.num_candidates)
            )
            best_actions = best_actions[
                torch.arange(flatten_batch_size, device=tensordict.device_safe()).unsqueeze(1), top_k
            ]
            actions_means = best_actions.mean(dim=1, keepdim=True)
            actions_stds = best_actions.std(dim=1, keepdim=True)
        return actions_means[:, :, 0].reshape(*batch_size, *self.action_spec.shape)


class PrecomputedActionsSequentialSetter:
    def __init__(self, actions):
        self.actions = actions
        self.cmpt = 0

    def __call__(self, td):
        if self.cmpt >= self.actions.shape[1]:
            raise ValueError("Precomputed actions are too short")
        td = td.set("action", self.actions[:, self.cmpt])
        self.cmpt += 1
        return td
