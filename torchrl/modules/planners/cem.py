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
    """
    CEMPlanner Module. This class inherits from TensorDictModule.
    
    Provided a TensorDict, this module will perform a CEM planning step.
    The CEM planning step is performed by sampling actions from a Gaussian distribution with zero mean and unit variance.
    The actions are then used to perform a rollout in the environment.
    The rewards are then used to update the mean and standard deviation of the Gaussian distribution.
    The mean and standard deviation of the Gaussian distribution are then used to sample actions for the next planning step.
    The CEM planning step is repeated for a specified number of steps.
    At the end, we recover the best action which is the one that maximizes the reward given a planning horizon.

    Args:
        env (Environment): The environment to perform the planning step on (Can be ModelBasedEnv or EnvBase).
        planning_horizon (int): The number of steps to perform the planning step for.
        optim_steps (int): The number of steps to perform the MPC planning step for.
        num_candidates (int): The number of candidates to sample from the Gaussian distribution.
        num_top_k_candidates (int): The number of top candidates to use to update the mean and standard deviation of the Gaussian distribution.
        reward_key (str): The key in the TensorDict to use to retrieve the reward.
        action_key (str): The key in the TensorDict to use to store the action.

    Returns:
        TensorDict: The TensorDict with the action added.
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

    def planning(self, td: TensorDictBase) -> torch.Tensor:
        batch_size = td.batch_size
        expanded_original_td = (
            td.expand(*batch_size, self.num_candidates).view(-1)
        )
        flatten_batch_size = batch_size.numel()
        actions_means = torch.zeros(
            flatten_batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
            device=td.device,
            dtype=self.env.action_spec.dtype,
        )
        actions_stds = torch.ones(
            flatten_batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
            device=td.device,
            dtype=self.env.action_spec.dtype,
        )
        for _ in range(self.optim_steps):
            actions = actions_means + actions_stds * torch.randn(
                flatten_batch_size,
                self.num_candidates,
                self.planning_horizon,
                *self.action_spec.shape,
                device=td.device,
                dtype=self.env.action_spec.dtype,
            )
            actions = actions.view(
                flatten_batch_size * self.num_candidates,
                self.planning_horizon,
                *self.action_spec.shape,
            )
            actions = self.env.action_spec.project(actions)
            optim_td = expanded_original_td.to_tensordict()
            policy = PrecomputedActionsSequentialSetter(actions)
            optim_td = self.env.rollout(
                max_steps=self.planning_horizon,
                policy=policy,
                auto_reset=False,
                tensordict=optim_td,
            )
            rewards = (
                optim_td.get(self.reward_key)
                .sum(dim=1)
                .reshape(flatten_batch_size, self.num_candidates)
            )
            _, top_k = rewards.topk(self.num_top_k_candidates, dim=1)
            top_k += (
                torch.arange(0, flatten_batch_size, device=td.device).unsqueeze(
                    1
                )
                * self.num_candidates
            )
            best_actions = actions.view(
                flatten_batch_size,
                self.num_candidates,
                self.planning_horizon,
                *self.action_spec.shape,
            )[torch.arange(flatten_batch_size), top_k]
            actions_means = best_actions.mean(dim=1, keepdim=True)
            actions_stds = best_actions.std(dim=1, keepdim=True)
        return (actions_means[:, :, 0]).view(*batch_size, *self.action_spec.shape)


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
