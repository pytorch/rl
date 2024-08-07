# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.envs.common import EnvBase
from torchrl.modules.planners.common import MPCPlannerBase


class MPPIPlanner(MPCPlannerBase):
    """MPPI Planner Module.

    Reference:
     - Model predictive path integral control using covariance variable importance
     sampling. (Williams, G., Aldrich, A., and Theodorou, E. A.) https://arxiv.org/abs/1509.01149
     - Temporal Difference Learning for Model Predictive Control
    (Hansen N., Wang X., Su H.) https://arxiv.org/abs/2203.04955

    This module will perform a MPPI planning step when given a TensorDict
    containing initial states.

    A call to the module returns the actions that empirically maximised the
    returns given a planning horizon

    Args:
        env (EnvBase): The environment to perform the planning step on (can be
            `ModelBasedEnv` or :obj:`EnvBase`).
        planning_horizon (int): The length of the simulated trajectories
        optim_steps (int): The number of optimization steps used by the MPC
            planner
        num_candidates (int): The number of candidates to sample from the
            Gaussian distributions.
        top_k (int): The number of top candidates to use to
            update the mean and standard deviation of the Gaussian distribution.
        reward_key (str, optional): The key in the TensorDict to use to
            retrieve the reward. Defaults to "reward".
        action_key (str, optional): The key in the TensorDict to use to store
            the action. Defaults to "action"

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Composite, Unbounded
        >>> from torchrl.envs.model_based import ModelBasedEnvBase
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import ValueOperator
        >>> from torchrl.objectives.value import TDLambdaEstimator
        >>> class MyMBEnv(ModelBasedEnvBase):
        ...     def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
        ...         super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
        ...         self.state_spec = Composite(
        ...             hidden_observation=Unbounded((4,))
        ...         )
        ...         self.observation_spec = Composite(
        ...             hidden_observation=Unbounded((4,))
        ...         )
        ...         self.action_spec = Unbounded((1,))
        ...         self.reward_spec = Unbounded((1,))
        ...
        ...     def _reset(self, tensordict: TensorDict) -> TensorDict:
        ...         tensordict = TensorDict(
        ...             {},
        ...             batch_size=self.batch_size,
        ...             device=self.device,
        ...         )
        ...         tensordict = tensordict.update(
        ...             self.full_state_spec.rand())
        ...         tensordict = tensordict.update(
        ...             self.full_action_spec.rand())
        ...         tensordict = tensordict.update(
        ...             self.full_observation_spec.rand())
        ...         return tensordict
        ...
        >>> from torchrl.modules import MLP, WorldModelWrapper
        >>> import torch.nn as nn
        >>> world_model = WorldModelWrapper(
        ...     TensorDictModule(
        ...         MLP(out_features=4, activation_class=nn.ReLU, activate_last_layer=True, depth=0),
        ...         in_keys=["hidden_observation", "action"],
        ...         out_keys=["hidden_observation"],
        ...     ),
        ...     TensorDictModule(
        ...         nn.Linear(4, 1),
        ...         in_keys=["hidden_observation"],
        ...         out_keys=["reward"],
        ...     ),
        ... )
        >>> env = MyMBEnv(world_model)
        >>> value_net = nn.Linear(4, 1)
        >>> value_net = ValueOperator(value_net, in_keys=["hidden_observation"])
        >>> adv = TDLambdaEstimator(
        ...     gamma=0.99,
        ...     lmbda=0.95,
        ...     value_network=value_net,
        ... )
        >>> # Build a planner and use it as actor
        >>> planner = MPPIPlanner(
        ...     env,
        ...     adv,
        ...     temperature=1.0,
        ...     planning_horizon=10,
        ...     optim_steps=11,
        ...     num_candidates=7,
        ...     top_k=3)
        >>> env.rollout(5, planner)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                hidden_observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        hidden_observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
    """

    def __init__(
        self,
        env: EnvBase,
        advantage_module: nn.Module,
        temperature: float,
        planning_horizon: int,
        optim_steps: int,
        num_candidates: int,
        top_k: int,
        reward_key: str = ("next", "reward"),
        action_key: str = "action",
    ):
        super().__init__(env=env, action_key=action_key)
        self.advantage_module = advantage_module
        self.planning_horizon = planning_horizon
        self.optim_steps = optim_steps
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.reward_key = reward_key
        self.register_buffer("temperature", torch.as_tensor(temperature))

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
        adv_topk_shape = (
            *batch_size,
            self.top_k,
            1,
            1,
        )
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
            # compute advantage
            self.advantage_module(optim_tensordict)
            # get advantage of the current state
            advantage = optim_tensordict["advantage"][..., :1, :]
            # get top-k trajectories
            _, top_k = advantage.topk(self.top_k, dim=K_DIM)
            # get omega weights for each top-k trajectory
            vals = advantage.gather(K_DIM, top_k.expand(adv_topk_shape))
            Omegas = (self.temperature * vals).exp()

            # gather best actions
            best_actions = actions.gather(K_DIM, top_k.expand(action_topk_shape))

            # compute weighted average
            _action_means = (Omegas * best_actions).sum(
                dim=K_DIM, keepdim=True
            ) / Omegas.sum(K_DIM, True)
            _action_stds = (
                (Omegas * (best_actions - _action_means).pow(2)).sum(
                    dim=K_DIM, keepdim=True
                )
                / Omegas.sum(K_DIM, True)
            ).sqrt()
            container.set_(("stats", "_action_means"), _action_means)
            container.set_(("stats", "_action_stds"), _action_stds)
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
