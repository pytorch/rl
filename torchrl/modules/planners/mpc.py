# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch
from torchrl.modules import TensorDictModule
from typing import Optional
from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.envs import EnvBase


class MPCPlanner(TensorDictModule):
    """
    MCPlanner Module. This class inherits from TensorDictModule.
    Provided a TensorDict, this module will perform a Model Predictive Control (MPC) planning step.
    The MPC planning step is performed by sampling actions from a Gaussian distribution with zero mean and unit variance.
    The actions are then used to perform a rollout in the environment.
    The rewards are then used to update the mean and standard deviation of the Gaussian distribution.
    The mean and standard deviation of the Gaussian distribution are then used to sample actions for the next planning step.
    The MPC planning step is repeated for a specified number of steps.
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
        out_keys = [action_key]
        in_keys = [env.input_spec.keys()]
        super().__init__(env, in_keys=in_keys, out_keys=out_keys)
        self.env = env
        self.action_spec = env.action_spec
        self.planning_horizon = planning_horizon
        self.optim_steps = optim_steps
        self.num_candidates = num_candidates
        self.num_top_k_candidates = num_top_k_candidates
        self.reward_key = reward_key

    def _call_module(self, tensordict):
        batch_size = tensordict.batch_size
        expanded_original_td = (
            tensordict.clone().expand(*batch_size, self.num_candidates).flatten()
        )
        flatten_batch_size = batch_size.numel()
        actions_means = torch.zeros(
            flatten_batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
            device=tensordict.device,
        )
        actions_stds = torch.ones(
            flatten_batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
            device=tensordict.device,
        )
        for _ in range(self.optim_steps):
            actions = actions_means + actions_stds * torch.randn(
                flatten_batch_size,
                self.num_candidates,
                self.planning_horizon,
                *self.action_spec.shape,
                device=tensordict.device,
            )
            actions = actions.view(
                flatten_batch_size * self.num_candidates,
                self.planning_horizon,
                *self.action_spec.shape,
            )
            optim_td = expanded_original_td.clone()
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
                torch.arange(0, flatten_batch_size, device=tensordict.device).unsqueeze(
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

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        if "params" in kwargs or "vmap" in kwargs:
            raise ValueError("params not supported")
        action = self._call_module(tensordict)
        tensors = (action,)
        tensordict_out = self._write_to_tensordict(
            tensordict,
            tensors,
            tensordict_out,
        )
        return tensordict_out


# class MCPCPlanner(nn.Module):
#     def __init__(
#         self,
#         env,
#         action_spec,
#         planning_horizon,
#         optim_steps,
#         num_candidates,
#         num_top_k_candidates,
#         reward_key="reward",
#     ):
#         super().__init__()
#         self.env = env
#         self.action_spec = action_spec
#         self.planning_horizon = planning_horizon
#         self.optim_steps = optim_steps
#         self.num_candidates = num_candidates
#         self.num_top_k_candidates = num_top_k_candidates
#         self.reward_key = reward_key

#     def forward(self, tensordict):
#         batch_size = tensordict.batch_size
#         expanded_original_td = (
#             tensordict.clone().expand(*batch_size, self.num_candidates).flatten()
#         )
#         flatten_batch_size = batch_size.numel()
#         actions_means = torch.zeros(
#             flatten_batch_size,
#             1,
#             self.planning_horizon,
#             *self.action_spec.shape,
#             device=tensordict.device
#         )
#         actions_stds = torch.ones(
#             flatten_batch_size,
#             1,
#             self.planning_horizon,
#             *self.action_spec.shape,
#             device=tensordict.device
#         )
#         for _ in range(self.optim_steps):
#             actions = actions_means + actions_stds * torch.randn(
#                 flatten_batch_size,
#                 self.num_candidates,
#                 self.planning_horizon,
#                 *self.action_spec.shape,
#                 device=tensordict.device
#             )
#             actions = actions.view(
#                 flatten_batch_size * self.num_candidates,
#                 self.planning_horizon,
#                 *self.action_spec.shape
#             )
#             optim_td = expanded_original_td.clone()
#             policy = PrecomputedActionsSequentialSetter(actions)
#             optim_td = self.env.rollout(
#                 max_steps=self.planning_horizon,
#                 policy=policy,
#                 auto_reset=False,
#                 tensordict=optim_td,
#             )
#             rewards = (
#                 optim_td.get(self.reward_key)
#                 .sum(dim=1)
#                 .reshape(flatten_batch_size, self.num_candidates)
#             )
#             _, top_k = rewards.topk(self.num_top_k_candidates, dim=1)
#             top_k += (
#                 torch.arange(0, flatten_batch_size, device=tensordict.device).unsqueeze(
#                     1
#                 )
#                 * self.num_candidates
#             )
#             best_actions = actions.view(
#                 flatten_batch_size,
#                 self.num_candidates,
#                 self.planning_horizon,
#                 *self.action_spec.shape
#             )[torch.arange(flatten_batch_size), top_k]
#             actions_means = best_actions.mean(dim=1, keepdim=True)
#             actions_stds = best_actions.std(dim=1, keepdim=True)
#         return (actions_means[:, :, 0]).view(*batch_size, *self.action_spec.shape)


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
