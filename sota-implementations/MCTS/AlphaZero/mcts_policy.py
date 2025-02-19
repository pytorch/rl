# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

# noinspection PyProtectedMember
from tensordict.nn.common import TensorDictModuleBase
from torch.distributions.dirichlet import _Dirichlet
from torchrl.envs import EnvBase
from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type

from torchrl.objectives.value.functional import reward2go

from torchrl.data import MCTSNode, MCTSChildren


@dataclass
class AlphaZeroConfig:
    num_simulations: int = 100
    simulation_max_steps: int = 20  # 30 for chess
    max_steps: int = 20
    c_puct: float = 1.0
    dirichlet_alpha: float | None = 0.03
    use_value_network: bool = False


class ActionExplorationModule:
    """An ActionExplorationModule is responsible for selecting an action for a given MCTSNode."""

    def __init__(
        self,
        action_key: NestedKey = "action",
    ):
        self.action_key = action_key

    def forward(self, node: MCTSNode) -> TensorDictBase:
        """Forward function for ActionExplorationModule.

        During exploration the forward method will select the action
        with the highest score in the MCTSNode.scores.

        If there are multiple actions with ``max(MCTSNode.scores)``, it will select one
        of those actions at random.

        During inference the forward method will select the action with the highest visit count.

        Args:
            node (MCTSNode): MCTSNode associated with the state

        Returns:
            TensorDictBase: The state associated with the MCTSNode along with the
            the selected action stored under action_key

        """
        tensordict = node.state.clone(False)

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            tensordict[self.action_key] = self.explore_action(node)
        elif exploration_type() in (ExplorationType.MODE, ExplorationType.DETERMINISTIC, ExplorationType.MEAN):
            tensordict[self.action_key] = self.get_greedy_action(node)

        return tensordict

    def get_greedy_action(self, node: MCTSNode) -> torch.Tensor:
        action = torch.argmax(node.children.visits)
        return action

    def explore_action(self, node: MCTSNode) -> torch.Tensor:
        action_scores = node.score

        max_value = torch.max(action_scores)
        action = torch.argmax(
            torch.rand_like(action_scores) * (action_scores == max_value)
        )
        # return torch.nn.functional.one_hot(action, action_value.shape[-1])
        return action


class BatchedActionExplorationModule(ActionExplorationModule):
    """Class responsible for selecting multiple unique actions for a given MCTSNode to enable batched exploration."""

    def __init__(
        self,
        action_key: NestedKey = "action",
        batch_size: int = 1,
    ):
        super().__init__(action_key)
        self.batch_size = batch_size

    def forward(self, node: MCTSNode) -> TensorDictBase:
        tensordict = node.state.clone(False)

        action_score = node.score
        _, actions = torch.topk(action_score, self.batch_size, dim=-1)

        tensordict = tensordict.expand(self.batch_size)
        tensordict[self.action_key] = actions.unsqueeze(-1)

        return tensordict

    def set_node(self, node: MCTSNode) -> None:
        self.node = node


class UpdateTreeStrategy:
    def __init__(
        self,
        value_network: TensorDictModuleBase,
        action_key: NestedKey = "action",
        use_value_network: bool = True,
    ):
        self.action_key = action_key
        self.value_network = value_network
        self.root: MCTSNode
        self.use_value_network = use_value_network

    def update(self, rollout: TensorDictBase) -> None:
        target_value = torch.zeros(rollout.batch_size[-1] + 1, dtype=torch.float32)
        done = torch.zeros_like(target_value, dtype=torch.bool)
        done[-1] = True
        if rollout[("next", "done")][-1]:
            target_value[-1] = rollout[("next", "reward")][-1]
        else:
            if self.use_value_network:
                target_value[-1] = self.value_network(rollout[-1]["next"])[
                    "state_value"
                ]
            else:
                target_value[-1] = 0

        target_value = reward2go(target_value, done, gamma=0.99, time_dim=-1)
        node = self.root
        for idx in range(rollout.batch_size[-1]):
            action = rollout[self.action_key][idx]
            node = node.get_child(action)
            node.value = (node.value * node.visits + target_value[idx]) / (
                node.visits + 1
            )
            node.visits += 1

    def start_simulation(self, device) -> None:
        self.root = MCTSNode.root().to(device)  # type: ignore


class ExpansionStrategy:
    """The rollout policy in expanding tree.

    This policy will use to initialize a node when it gets expanded at the first time.
    """

    def forward(self, node: MCTSNode) -> MCTSNode:
        """The node to be expanded.

        The output Tensordict will be used in future to select action.

        Args:
            node (MCTSNode): The state that need to be explored

        Returns:
            A initialized statistics to select actions in the future.

        """

        if not node.expanded:
            self.expand(node)

        return node

    @abstractmethod
    def expand(self, node: MCTSNode) -> None:
        ...

    def set_node(self, node: MCTSNode) -> None:
        self.node = node


class BatchedRootExpansionStrategy(ExpansionStrategy):
    def __init__(
        self,
        policy_module: TensorDictModule,
        module_action_value_key: NestedKey = "action_value",
    ):
        super().__init__()
        assert module_action_value_key in policy_module.out_keys
        self.policy_module = policy_module
        self.action_value_key = module_action_value_key

    def expand(self, node: MCTSNode) -> None:
        policy_netword_td = node.state.select(*self.policy_module.in_keys)
        policy_netword_td = self.policy_module(policy_netword_td)
        p_sa = policy_netword_td[self.action_value_key]
        node.children = MCTSChildren.init_from_prob(p_sa)
        # setattr(node, "truncated", torch.ones(1, dtype=torch.bool))


class AlphaZeroExpansionStrategy(ExpansionStrategy):
    def __init__(
        self,
        policy_module: TensorDictModule,
        module_action_value_key: NestedKey = "action_value",
    ):
        super().__init__()
        assert module_action_value_key in policy_module.out_keys
        self.policy_module = policy_module
        self.action_value_key = module_action_value_key

    def expand(self, node: MCTSNode) -> None:
        policy_netword_td = node.state.select(*self.policy_module.in_keys)
        policy_netword_td = self.policy_module(policy_netword_td)
        p_sa = policy_netword_td[self.action_value_key]
        node.children.priors = p_sa  # prior_action_value
        node.children.vals = torch.zeros_like(p_sa)  # action_value
        node.children.visits = torch.zeros_like(p_sa)  # action_count
        # setattr(node, "truncated", torch.ones(1, dtype=torch.bool))


class PUCTSelectionPolicy:
    """Predictor-Upper Confidence.

    Confidence applied to Trees was proposed in the AlphaZero
    strikes a balance between exploration of unvisited states, probabilities from a policy
    network and exploiting values within a tree search. A large `cpuct` promotes exploration
    over exploitation.
    """

    def __init__(
        self,
        cpuct: float = 1.0,
    ):
        self.cpuct = cpuct
        self.node: MCTSNode

    def forward(self, node: MCTSNode) -> MCTSNode:
        n = torch.sum(node.children.visits, dim=-1) + 1
        u_sa = (
            self.cpuct
            * node.children.priors
            * torch.sqrt(n)
            / (1 + node.children.visits)
        )

        optimism_estimation = node.children.vals + u_sa
        node.scores = optimism_estimation

        return node


class DirichletNoiseModule:
    """A module that injects some noise in the root node of a Monte-Carlo Tree to promote exploration."""

    def __init__(
        self,
        alpha: float = 0.3,
        epsilon: float = 0.25,
    ):
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, node: MCTSNode) -> MCTSNode:
        if node.children.priors.device.type == "mps":
            device = node.children.priors.device
            noise = _Dirichlet.apply(
                self.alpha * torch.ones_like(node.children.priors).cpu()
            )
            noise = noise.to(device)  # type: ignore
        else:
            noise = _Dirichlet.apply(self.alpha * torch.ones_like(node.children.priors))

        noisy_priors = (1 - self.epsilon) * node.children.priors + self.epsilon * noise  # type: ignore
        node.children.priors = noisy_priors
        return node


class MCTSPolicy(TensorDictModuleBase):
    """An implementation of MCTS algorithm.

    Args:
        expansion_strategy: a policy to initialize stats of a node at its first visit.
        selection_strategy: a policy to select action in each state
        exploration_strategy: a policy to exploration vs exploitation
    """

    node: MCTSNode

    def __init__(
        self,
        expansion_strategy: AlphaZeroExpansionStrategy,
        selection_strategy: PUCTSelectionPolicy | None= None,
        exploration_strategy: ActionExplorationModule |None= None,
        batch_size: int = 1,
    ):
        if selection_strategy is None:
            selection_strategy = PUCTSelectionPolicy()

        if expansion_strategy is None:
            expansion_strategy = ActionExplorationModule()

        super().__init__(
            in_keys=expansion_strategy.policy_module.in_keys,
            out_keys=exploration_strategy.action_key,
        )
        self.expansion_strategy = expansion_strategy
        self.selection_strategy = selection_strategy
        self.exploration_strategy = exploration_strategy
        self.batch_size = batch_size

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not hasattr(self, "node"):
            raise RuntimeError("the MCTS policy has not been initialized. Please provide a node through policy.set_node().")
        if not self.node.expanded:
            self.node.state = tensordict  # type: ignore
        self.expansion_strategy.forward(self.node)
        self.selection_strategy.forward(self.node)
        tensordict = self.exploration_strategy.forward(self.node)

        batched_nodes = []
        if self.batch_size > 1:
            for i in range(self.batch_size):
                node: MCTSNode = self.node[i]  # type: ignore
                if not tensordict[i]["terminated"]:
                    node = node.get_child(tensordict[i]["action"])
                batched_nodes.append(node)
            self.set_node(torch.stack(batched_nodes))  # type: ignore
        else:
            self.set_node(self.node.get_child(tensordict["action"]))

        return tensordict

    def set_node(self, node: MCTSNode) -> None:
        self.node = node


class SimulatedSearchPolicy(TensorDictModuleBase):
    """A simulated search policy.

    In each step, it simulates `n` rollout of maximum steps of `max_simulation_steps`
    using the given policy and then choose the best action given the simulation results.

    Args:
        policy: a policy to select action in each simulation rollout.
        env: an environment to simulate a rollout
        num_simulation: the number of simulations before choosing an action
        simulation_max_steps: the max steps of each simulated rollout
        max_steps: the max steps performed by SimulatedSearchPolicy
        noise_module: a module to inject noise in the root node for exploration
    """

    def __init__(
        self,
        policy: MCTSPolicy,
        tree_updater: UpdateTreeStrategy,
        env: EnvBase,
        num_simulations: int,
        simulation_max_steps: int,
        max_steps: int,
        noise_module: DirichletNoiseModule | None = None,
    ):
        if noise_module is None:
            noise_module = DirichletNoiseModule()
        self.in_keys = policy.in_keys
        self.out_keys = policy.out_keys

        super().__init__()
        self.policy = policy
        self.tree_updater = tree_updater
        self.env = env
        self.num_simulations = num_simulations
        self.simulation_max_steps = simulation_max_steps
        self.max_steps = max_steps
        self.noise_module = noise_module
        self.root_list: List[MCTSNode] = []
        self.init_state: TensorDict
        self._steps = 0

    def forward(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)
        self._steps += 1

        with torch.no_grad():
            self.start_simulation(tensordict)

            with set_exploration_type(ExplorationType.RANDOM):
                for _ in range(self.num_simulations):
                    self.simulate()

            with set_exploration_type(ExplorationType.MODE):
                root = self.tree_updater.root
                tensordict = self.policy.exploration_strategy.forward(root)
                self.root_list.append(root)

            # This can be achieved with step counter
            if self._steps == self.max_steps:
                tensordict["truncated"] = torch.ones(
                    (1), dtype=torch.bool, requires_grad=False, device=tensordict.device
                )

            return tensordict

    def simulate(self) -> None:
        self.initialize_policy_nodes()

        rollout = self.env.rollout(
            max_steps=self.simulation_max_steps,
            policy=self.policy,
            return_contiguous=False,
        )

        # Resets the environment to the original state # type: ignore
        self.env.set_state(self.init_state.clone(True))  # type: ignore

        # update the nodes visited during the simulation
        self.tree_updater.update(rollout)  # type: ignore

    def start_simulation(self, tensordict: TensorDictBase) -> None:
        # creates new root node for the MCTS tree
        self.tree_updater.start_simulation(tensordict.device)

        # make a copy of the initial state
        self.init_state = self.env.copy_state()

        # initialize and expand the root
        self.tree_updater.root.state = tensordict
        self.policy.expansion_strategy.forward(self.tree_updater.root)

        # inject dirichlet noise for exploration
        if self.noise_module is not None:
            self.noise_module.forward(self.tree_updater.root)

    def initialize_policy_nodes(self) -> None:
        # reset the policy node to the root
        if self.policy.batch_size > 1:
            self.policy.set_node(self.tree_updater.root.expand(self.policy.batch_size))  # type: ignore
        else:
            self.policy.set_node(self.tree_updater.root)
