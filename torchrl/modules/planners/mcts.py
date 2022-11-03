# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import math
from typing import Optional, Union
from warnings import warn

import torch
from torch import nn
from torch.distributions.dirichlet import _Dirichlet

from torchrl.data import DEVICE_TYPING
from torchrl.data.tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

# This file is inspired by https://github.com/FelixOpolka/Single-Player-MCTS
# Refactoring involves: renaming, torch and tensordict compatibility and integration in torchrl's API
from torchrl.envs.utils import step_mdp
from torchrl.modules import TensorDictModule


class MCTSPlanner:
    """To do."""

    pass


class _Children(collections.UserDict):
    """Children class for _MCTSNode.

    When updating the children of a node, we will also write their tensordict
    in the node's state tensordict.

    """

    def __init__(self, tensordict, env, current, **kwargs):
        self._tensordict = tensordict
        super().__init__(**kwargs)
        children = self._tensordict.get(
            "_children",
            lambda: TensorDict(
                {}, torch.Size([]), device=self._tensordict.device, _run_checks=False
            ),
        )
        # no op if key was already present
        self._tensordict["_children"] = children
        if len(children.keys()):
            # recreates the tree if needed
            # we can already populate the children dict
            for k in children.keys():
                child_node = _MCTSNode.make_child_node_from_tensordict(
                    children[k], env=env, parent=current, prev_action=int(k)
                )
                collections.UserDict.__setitem__(self, k, child_node)

    def __setitem__(self, key: int, value: _MCTSNode):
        super().__setitem__(key, value)
        children = self._tensordict.get("_children")
        children[str(key)] = value.state


class _MCTSNode:
    """Represents a node in the Monte-Carlo search tree. Each node holds a single environment state.

    Reference: https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf

    Args:
        state (TensorDictBase): A tensordict representing the state of the node.
        n_actions (int): number of actions available at that stage.
        env (EnvBase): a stateless environment reading a state and an action
            through a step method.
        parent (_MCTSNode): a parent node.
        prev_action (int): the action that lead to this node.
        exploration_factor (float, optional): Exploration constant. Default: :obj:`1.38`.
        d_noise_alpha (float, optional): Dirichlet noise alpha parameter. Default: :obj:`0.03`.
        temp_threshold (int, optional): Number of steps into the episode after
            which we always select the action with highest action probability
            rather than selecting randomly.

    """

    def __init__(
        self,
        state: TensorDictBase,
        n_actions: int,
        env: EnvBase,
        parent: _MCTSNode,
        prev_action: int,
        exploration_factor: float = 1.38,
        d_noise_alpha: float = 0.03,
        temp_threshold: int = 5,
        device: Optional[DEVICE_TYPING] = None,
    ):
        self.state = state
        self.n_actions = n_actions
        self.env = env
        self._children = None

        if parent is None:
            self.depth = 0
            parent = _VoidNode()
        else:
            self.depth = parent.depth + 1
        self.parent = parent
        self.prev_action = prev_action
        self.exploration_factor = exploration_factor
        self.d_noise_alpha = d_noise_alpha
        self.temp_threshold = temp_threshold

        self._device = device
        state_keys = set(state.keys())
        if "_n_vlosses" not in state_keys:
            self.state["_n_vlosses"] = torch.zeros(
                1, device=self.device, dtype=torch.long
            )
        if "n_actions" not in state_keys:
            self.state["n_actions"] = torch.tensor(
                [n_actions], dtype=torch.long, device=self.device
            )
        if "_child_visit_count" not in state_keys:
            self.state["_child_visit_count"] = torch.zeros(
                [n_actions], dtype=torch.long, device=self.device
            )
        if "_is_expanded" not in state_keys:
            self.state["_is_expanded"] = torch.zeros(
                [1], dtype=torch.bool, device=self.device
            )
        if "_child_total_value" not in state_keys:
            self.state["_child_total_value"] = torch.zeros(
                [n_actions], device=self.device
            )
        # Save copy of original prior before it gets mutated by dirichlet noise
        if "_original_prior" not in state_keys:
            self.state["_original_prior"] = torch.zeros([n_actions], device=self.device)
        if "_child_prior" not in state_keys:
            self.state["_child_prior"] = torch.zeros([n_actions], device=self.device)
        if hasattr(parent, "_children") and parent._children is not None:
            parent._children[prev_action] = self
        self._children = _Children(self.state, self.env, self)

    @staticmethod
    def make_child_node_from_tensordict(tensordict, env, parent, prev_action):
        return _MCTSNode(
            tensordict,
            n_actions=tensordict["n_actions"].item(),
            env=env,
            parent=parent,
            prev_action=prev_action,
            # TODO: pass hyperparams
            device=tensordict.device,
        )

    @property
    def _n_vlosses(self):
        return self.state.get("_n_vlosses")

    @_n_vlosses.setter
    def _n_vlosses(self, value):
        self.state.set("_n_vlosses", value)

    @property
    def depth(self) -> int:
        return self.state.get("_depth").item()

    @depth.setter
    def depth(self, depth: int):
        self.state.set("_depth", torch.tensor([depth], dtype=torch.long))

    @property
    def children(self):
        if self._children is None:
            self._children = _Children(self.state, self.env, self)
        return self._children

    @property
    def _child_visit_count(self) -> torch.Tensor:
        return self.state.get("_child_visit_count")

    @_child_visit_count.setter
    def _child_visit_count(self, value):
        self.state.set("_child_visit_count", value)

    @property
    def _child_total_value(self) -> torch.Tensor:
        return self.state.get("_child_total_value")

    @_child_total_value.setter
    def _child_total_value(self, value):
        self.state.set("_child_total_value", value)

    @property
    def action_log_prob(self) -> torch.Tensor:
        return self.state.get("action_log_prob")

    @action_log_prob.setter
    def action_log_prob(self, value):
        self.state.set("action_log_prob", value)

    @property
    def value(self) -> torch.Tensor:
        return self.state.get("value")

    @value.setter
    def value(self, value):
        self.state.set("value", value)

    @property
    def _original_prior(self) -> torch.Tensor:
        return self.state.get("_original_prior")

    @_original_prior.setter
    def _original_prior(self, value):
        self.state.set("_original_prior", value)

    @property
    def _child_prior(self) -> torch.Tensor:
        return self.state.get("_child_prior")

    @_child_prior.setter
    def _child_prior(self, value):
        self.state.set("_child_prior", value)

    @property
    def is_expanded(self):
        """Boolean value that is set to True once a node has been identified as a leaf and `node.incorporate_estimates` has been called."""
        return self.state.get("_is_expanded")

    @is_expanded.setter
    def is_expanded(self, value: torch.Tensor):
        self.state.set("_is_expanded", value)

    @property
    def device(self):
        return self._device

    @property
    def visit_count(self) -> int:
        """Number of times this particular node has been accessed."""
        return self.parent._child_visit_count[self.prev_action]

    @visit_count.setter
    def visit_count(self, value):
        self.parent._child_visit_count[self.prev_action] = value

    @property
    def total_value(self) -> torch.Tensor:
        return self.parent._child_total_value[self.prev_action]

    @total_value.setter
    def total_value(self, value):
        self.parent._child_total_value[self.prev_action] = value

    @property
    def action_value(self) -> torch.Tensor:
        return self.total_value / (1 + self.visit_count)

    @property
    def exploration_credit(self) -> torch.Tensor:
        """Exploration boost factor: gives a high credit to moves with few simulations."""
        return (
            self.exploration_factor
            * math.sqrt(1 + self.visit_count)
            * self._child_prior
            / (1 + self._child_visit_count)
        )

    @property
    def exploitation_credit(self) -> torch.Tensor:
        """Exploration credit: gives a high credit to moves with a high average value."""
        return self._child_total_value / (1 + self._child_visit_count)

    @property
    def action_score(self) -> torch.Tensor:
        """Action score.

        Proposed in: An Adaptive Sampling Algorithm for Solving Markov Decision
        Processes. Hyeong Soo Chang, Michael C. Fu, Jiaqiao Hu, Steven I. Marcus,
        https://doi.org/10.1287/opre.1040.0145

        """
        action_score = self.exploitation_credit + self.exploration_credit
        self.state.set("action_score", action_score)
        return action_score

    def select_leaf(self) -> _MCTSNode:
        """Finds a leaf in the MCT rooted in the current node.

        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        _child_visit_count and _child_total_value values but not as a dedicated
        node in the parent's children-mapping). Nodes are selected according to
        their action_score.

        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.

        """
        current = self
        while True:
            current.visit_count += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Choose action with highest score.
            best_move = current.action_score.argmax(-1)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action: int):
        """Adds a child node for the given action if it does not yet exists, and returns it.

        Args:
            action (int): Action to take in current state which leads to desired child node.

        Returns: a child _MCTSNode.
        """
        action = int(action)
        if action not in self.children:
            # Obtain state following given action.
            state = self.state.clone(recurse=False)
            new_state = state.set("action", torch.tensor([action]))
            action_hash = self.env._hash_action(new_state["action"])
            new_state = self.env.step(new_state)
            new_state = new_state["_children", action_hash]
            self.children[action] = _MCTSNode(
                new_state, self.n_actions, self.env, prev_action=action, parent=self
            )
        return self.children[action]

    def incorporate_estimates(
        self, action_log_probs: torch.Tensor, value: float, up_to: _MCTSNode
    ):
        """Method to be called if the node has just been expanded via `select_leaf`.

        It incorporates the prior action probabilities and state value estimated
        by the neural network.

        Args:
            action_log_probs (torch.Tensor): Action log-probabilities for the current
                node's state predicted by the neural network.
            value (float): Value of the current node's state predicted
                by the neural network.
            up_to (int): The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self._original_prior = self._child_prior = action_log_probs.exp()
        self._child_total_value = (
            torch.ones([self.n_actions], dtype=torch.float32, device=self.device)
            * value
        )
        self.backup_value(value, up_to=up_to)

        self.value = value
        self.action_log_prob = action_log_probs

    def revert_visits(self, up_to: _MCTSNode):
        """Revert visit increments.

        Sometimes, repeated calls to :doc:`select_leaf` return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But :doc:`select_leaf` has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.

        Args:
            up_to (_MCTSNode): the last node to call :obj:`revert_visit`.
        """
        self.visit_count -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def add_virtual_loss(self, up_to: _MCTSNode):
        """Propagate a virtual loss up to a given node.

        Args:
            up_to (_MCTSNode): The node to propagate until.
        """
        self._n_vlosses += 1
        self._child_total_value -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to: _MCTSNode):
        """Undo adding virtual loss.

        Args:
            up_to (_MCTSNode): The node to propagate until.
        """
        self._n_vlosses -= 1
        self._child_total_value += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def backup_value(self, value: Union[torch.Tensor, float], up_to: _MCTSNode):
        """Propagates a value estimation up to the root node.

        Args:
            value (torch.Tensor): Value estimate to be propagated.
            up_to (_MCTSNode): The node to propagate until.
        """
        if isinstance(value, torch.Tensor):
            value = value.squeeze()
        self.total_value += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self) -> torch.Tensor:
        return self.state["prev_done"]

    def inject_noise(self):
        dirch = _Dirichlet.apply(
            self.d_noise_alpha * torch.ones(self.n_actions, device=self.device)
        )
        self._child_prior = self._child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash: bool = False) -> torch.Tensor:
        """Returns the child visit counts as a probability distribution.

        Args:
            squash (bool): If :obj:`True`, exponentiates the probabilities by a temperature
                slightly larger than 1 to encourage diversity in early steps. Default: :obj:`False`.
        """
        probs = self._child_visit_count
        if squash:
            probs = probs ** 0.95
        return probs / probs.sum(-1, True)

    def get_return(self, discount: float = 1.0):
        """Returns the discounted total reward from the trajectory up to this node."""
        cur_node = self
        depth = self.depth
        total_reward = 0
        while depth > -1:
            if depth > 0:
                total_reward = discount * total_reward
                reward = cur_node.state["prev_reward"]
                total_reward = total_reward + reward
            cur_node = cur_node.parent
            depth = cur_node.depth
        return total_reward


class _VoidNode:
    """Special node that is used as the node above the initial root node to prevent having to deal with special cases when traversing the tree."""

    def __init__(self):
        self.parent = None
        self._child_visit_count = collections.defaultdict(float)
        self._child_total_value = collections.defaultdict(float)
        self.children = {}

    @property
    def depth(self):
        return -1

    def revert_virtual_loss(self, up_to=None):
        pass

    def add_virtual_loss(self, up_to=None):
        pass

    def revert_visits(self, up_to=None):
        pass

    def backup_value(self, value, up_to=None):
        pass


class MCTSPolicy(nn.Module):
    """A Monte-Carlo policy.

    Args:
        agent_network (TensorDictModule): Network for predicting action probabilities and
            state value estimate.
        env (EnvBase): A stateless env that defines environment dynamics (i.e. a simulator).
        simulations_per_move (int, optional): Number of traversals through the tree
            before performing a step. Default: 800.
        num_parallel (int, optional): Number of leaf nodes to collect before evaluating
            them in conjunction. Default: 8.
    """

    def __init__(
        self,
        agent_network: TensorDictModule,
        env: EnvBase,
        simulations_per_move: int = 800,
        num_parallel=8,
        temp_threshold=5,
    ):
        super().__init__()
        self.agent_network = agent_network
        # self.env = env
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = temp_threshold
        self.env = env

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self._root: Optional[_MCTSNode] = None

        self.n_actions = self.env.action_spec.space.n

    # Create a tree from the tensordict, if not yet present (or not the one expected).
    def _get_root(self, tensordict):
        self._root = _MCTSNode(
            tensordict,
            n_actions=self.n_actions,
            env=self.env,
            parent=None,
            prev_action=None,
        )
        return self._root

    def _root_and_tensordict_differ(self, root, tensordict):
        keys1 = {k for k in root.state.keys() if not k.startswith("_")}
        keys2 = {k for k in tensordict.keys() if not k.startswith("_")}
        keys = keys1.intersection(keys2)
        state_filter = root.state.select(*keys)
        tensordict = tensordict.select(*keys)
        return not (tensordict == state_filter).all()

    def _reset_root(self, tensordict):
        if self._root is not None and not self._root_and_tensordict_differ(
            self._root, tensordict
        ):
            print("keeping root")
            # then the root is right
            return self._root
        else:
            print("getting root")
            # the root has to be reset to the desired node
            return self._get_root(tensordict)

    def forward(self, tensordict):
        self._reset_root(tensordict)

        # inject noise
        self._root.inject_noise()

        # current sim
        current_simulations = self._root.visit_count

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while self._root.visit_count < current_simulations + self.simulations_per_move:
            self._tree_search()

        # Picks an action to execute in the environment.
        if self._root.depth > self.temp_threshold:
            # the visit count depends on the move score, hence more visited nodes have a higher score
            action = self._root._child_visit_count.argmax(-1)
        else:
            cdf = self._root._child_visit_count.cumsum(-1)
            cdf = cdf / cdf[..., -1:]
            selection = torch.rand_like(cdf[..., :1])
            action = torch.searchsorted(cdf, selection)
        tensordict["action"] = action
        return tensordict

    def _evaluate_leaf_states(self, leaves):
        leaf_states = torch.stack(
            [_select_public(leaf.state) for leaf in leaves], 0
        ).contiguous()
        leaf_states = self.agent_network(leaf_states)

        for leaf, leaf_state in zip(leaves, leaf_states):
            leaf.revert_virtual_loss(up_to=self._root)
            action_log_prob = leaf_state["action_log_prob"]
            value = leaf_state["state_value"]
            leaf.incorporate_estimates(action_log_prob, value, up_to=self._root)

    def _tree_search(self):
        """Performs multiple simulations in the tree (following trajectories) until a given amount of leaves to expand have been encountered.

        Then it expands and evaluates these leaf nodes.

        """
        leaves = []
        # Failsafe for when we encounter almost only done-states which would
        # prevent the loop from ever ending.
        failsafe = 0
        while len(leaves) < self.num_parallel and failsafe < self.num_parallel * 2:
            failsafe += 1
            leaf = self._root.select_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                value = leaf.get_return()
                leaf.backup_value(value, up_to=self._root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self._root)
            leaves.append(leaf)
        if failsafe == self.num_parallel * 2:
            warn("broke because of failsafe value")
        # print("len leaves:", len(leaves), "depths:", [leaf.depth for leaf in leaves], "num unique:", len(set(leaves)))
        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            self._evaluate_leaf_states(leaves)
        return leaves


def _select_public(tensordict: TensorDictBase) -> TensorDictBase:
    """Selects the public keys from a tensordict."""
    selected_keys = {key for key in tensordict.keys() if not key.startswith("_")}
    return tensordict.select(*selected_keys)
