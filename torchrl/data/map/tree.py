# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import weakref
from collections import deque

from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from tensordict import (
    merge_tensordicts,
    NestedKey,
    TensorClass,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from torchrl.data.map.tdstorage import TensorDictMap
from torchrl.data.map.utils import _plot_plotly_box, _plot_plotly_tree
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.data.tensor_specs import Composite

from torchrl.envs.common import EnvBase


class Tree(TensorClass["nocast"]):
    """Representation of a single MCTS (Monte Carlo Tree Search) Tree.

    This class encapsulates the data and behavior of a tree node in an MCTS algorithm.
    It includes attributes for storing information about the node, such as its children,
    visit count, and rollout data. Methods are provided for traversing the tree,
    computing statistics, and visualizing the tree structure.

    It is somewhat indistinguishable from a node or a vertex - we use the term "Tree" when talking about
    a node with children, "node" or "vertex" when talking about a place in the tree where a branching occurs.
    A node in the tree is defined primarily by its ``hash`` value. Usually, a ``hash`` is determined by a unique
    combination of state (or observation) and action. If one observation (found in the ``node`` attribute) has more than
    one action associated, each branch will be stored in the ``subtree`` attribute as a stack of ``Tree`` instances.

    Attributes:
        count (int): The number of visits to this node.
        index (torch.Tensor): Indices of the child nodes in the data map.
        hash (torch.Tensor): A hash value for this node.
            It may be the case that ``hash`` is ``None`` in the specific case where the root of the tree
            has more than one action associated. In that case, each subtree branch will have a different action
            associated and a hash correspoding to the ``(observation, action)`` pair.
        node_id (int): A unique identifier for this node.
        rollout (TensorDict): Rollout data following the observation encoded in this node, in a TED format.
            If there are multiple actions taken at this node, subtrees are stored in the corresponding
            entry. Rollouts can be reconstructed using the :meth:`~.rollout_from_path` method.
        node (TensorDict): Data defining this node (e.g., observations) before the next branching.
            Entries usually matches the ``in_keys`` in ``MCTSForest.node_map``.
        subtree (Tree): A stack of subtrees produced when actions are taken.
        num_children (int): The number of child nodes (read-only).
        is_terminal (bool): whether the tree has children nodes (read-only).
            If the tree is compact, ``is_terminal == True`` means that there are more than one child node in
            ``self.subtree``.

    Methods:
        __contains__: Whether another tree can be found in the tree.
        vertices: Returns a dictionary containing all vertices in the tree. Keys must be paths, ids or hashes.
        num_vertices: Returns the total number of vertices in the tree, with or without duplicates.
        edges: Returns a list of edges in the tree.
        valid_paths: Yields all valid paths in the tree.
        max_length: Returns the maximum length of any path in the tree.
        rollout_from_path: Reconstructs a rollout from a given path.
        plot: Visualizes the tree using a specified backend and figure type.
        get_node_by_id: returns the vertex given by its id in the tree.
        get_node_by_hash: returns the vertex given by its hash in the forest.

    """

    count: int | torch.Tensor = None
    wins: int | torch.Tensor = None

    index: torch.Tensor | None = None
    # The hash is None if the node has more than one action associated
    hash: int | None = None
    node_id: int | None = None

    # rollout following the observation encoded in node, in a TorchRL (TED) format
    rollout: TensorDict | None = None

    # The data specifying the node (typically an observation or a set of observations)
    node_data: TensorDict | None = None

    # Stack of subtrees. A subtree is produced when an action is taken.
    subtree: "Tree" = None

    # weakrefs to the parent(s) of the node
    _parent: weakref.ref | List[weakref.ref] | None = None

    # Specs: contains information such as action or observation keys and spaces.
    #  If present, they should be structured like env specs are:
    #  Composite(input_spec=Composite(full_state_spec=..., full_action_spec=...),
    #            output_spec=Composite(full_observation_spec=..., full_reward_spec=..., full_done_spec=...))
    #  where every leaf component is optional.
    specs: Composite | None = None

    @classmethod
    def make_node(
        cls,
        data: TensorDictBase,
        *,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        specs: Composite | None = None,
    ) -> Tree:
        """Creates a new node given some data."""
        if "next" in data.keys():
            rollout = data
            if not rollout.ndim:
                rollout = rollout.unsqueeze(0)
            subtree = TensorDict.lazy_stack([cls.make_node(data["next"][..., -1])])
        else:
            rollout = None
            subtree = None
        if device is None:
            device = data.device
        return cls(
            count=torch.zeros(()),
            wins=torch.zeros(()),
            node=data.exclude("action", "next"),
            rollout=rollout,
            subtree=subtree,
            device=device,
            batch_size=batch_size,
        )

    # Specs
    @property
    def full_observation_spec(self):
        """The observation spec of the tree.

        This is an alias for `Tree.specs['output_spec', 'full_observation_spec']`.
        """
        return self.specs["output_spec", "full_observation_spec"]

    @property
    def full_reward_spec(self):
        """The reward spec of the tree.

        This is an alias for `Tree.specs['output_spec', 'full_reward_spec']`.
        """
        return self.specs["output_spec", "full_reward_spec"]

    @property
    def full_done_spec(self):
        """The done spec of the tree.

        This is an alias for `Tree.specs['output_spec', 'full_done_spec']`.
        """
        return self.specs["output_spec", "full_done_spec"]

    @property
    def full_state_spec(self):
        """The state spec of the tree.

        This is an alias for `Tree.specs['input_spec', 'full_state_spec']`.
        """
        return self.specs["input_spec", "full_state_spec"]

    @property
    def full_action_spec(self):
        """The action spec of the tree.

        This is an alias for `Tree.specs['input_spec', 'full_action_spec']`.
        """
        return self.specs["input_spec", "full_action_spec"]

    @property
    def selected_actions(self) -> torch.Tensor | TensorDictBase | None:
        """Returns a tensor containing all the selected actions branching out from this node."""
        if self.subtree is None:
            return None
        return self.subtree.rollout[..., 0]["action"]

    @property
    def prev_action(self) -> torch.Tensor | TensorDictBase | None:
        """The action undertaken just before this node's observation was generated.

        Returns:
            a tensor, tensordict or None if the node has no parent.

        .. seealso:: This will be equal to :class:`~torchrl.data.Tree.branching_action` whenever the rollout data contains a single step.

        .. seealso:: :class:`All actions associated with a given node (or observation) in the tree <~torchrl.data.Tree.selected_action>`.

        """
        if self.rollout is None:
            return None
        return self.rollout[..., -1]["action"]

    @property
    def branching_action(self) -> torch.Tensor | TensorDictBase | None:
        """Returns the action that branched out to this particular node.

        Returns:
            a tensor, tensordict or None if the node has no parent.

        .. seealso:: This will be equal to :class:`~torchrl.data.Tree.prev_action` whenever the rollout data contains a single step.

        .. seealso:: :class:`All actions associated with a given node (or observation) in the tree <~torchrl.data.Tree.selected_action>`.

        """
        if self.rollout is None:
            return None
        return self.rollout[..., 0]["action"]

    @property
    def node_observation(self) -> torch.Tensor | TensorDictBase:
        """Returns the observation associated with this particular node.

        This is the observation (or bag of observations) that defines the node before a branching occurs.
        If the node contains a :attr:`~.rollout` attribute, the node observation is typically identical to the
        observation resulting from the last action undertaken, i.e., ``node.rollout[..., -1]["next", "observation"]``.

        If more than one observation key is associated with the tree specs, a :class:`~tensordict.TensorDict` instance
        is returned instead.

        For a more consistent representation, see :attr:`~.node_observations`.

        """
        # TODO: implement specs
        return self.node_data["observation"]

    @property
    def node_observations(self) -> torch.Tensor | TensorDictBase:
        """Returns the observations associated with this particular node in a TensorDict format.

        This is the observation (or bag of observations) that defines the node before a branching occurs.
        If the node contains a :attr:`~.rollout` attribute, the node observation is typically identical to the
        observation resulting from the last action undertaken, i.e., ``node.rollout[..., -1]["next", "observation"]``.

        If more than one observation key is associated with the tree specs, a :class:`~tensordict.TensorDict` instance
        is returned instead.

        For a more consistent representation, see :attr:`~.node_observations`.

        """
        # TODO: implement specs
        return self.node_data.select("observation")

    @property
    def visits(self) -> int | torch.Tensor:
        """Returns the number of visits associated with this particular node.

        This is an alias for the :attr:`~.count` attribute.

        """
        return self.count

    @visits.setter
    def visits(self, count):
        self.count = count

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "subtree" and value is not None:
            wr = weakref.ref(self._tensordict)
            if value._parent is None:
                value._parent = wr
            elif isinstance(value._parent, list):
                value._parent.append(wr)
            else:
                value._parent = [value._parent, wr]
        return super().__setattr__(name, value)

    @property
    def parent(self) -> Tree | None:
        """The parent of the node.

        If the node has a parent and this object is still present in the python workspace, it will be returned by this
        property.

        For re-branching trees, this property may return a stack of trees where every index of the stack corresponds to
        a different parent.

        .. note:: the ``parent`` attribute will match in content but not in identity: the tensorclass object is recustructed
            using the same tensors (i.e., tensors that point to the same memory locations).

        Returns:
            A ``Tree`` containing the parent data or ``None`` if the parent data is out of scope or the node is the root.
        """
        parent = self._parent
        if parent is not None:
            # Check that all parents match
            queue = [parent]

            def maybe_flatten_list(maybe_nested_list):
                if isinstance(maybe_nested_list, list):
                    for p in maybe_nested_list:
                        if isinstance(p, list):
                            queue.append(p)
                        else:
                            yield p()
                else:
                    yield maybe_nested_list()

            parent_result = None
            while len(queue):
                local_result = None
                for r in maybe_flatten_list(queue.pop()):
                    if local_result is None:
                        local_result = r
                    elif r is not None and r is not local_result:
                        if isinstance(local_result, list):
                            local_result.append(r)
                        else:
                            local_result = [local_result, r]
                if local_result is None:
                    continue
                # replicate logic at macro level
                if parent_result is None:
                    parent_result = local_result
                else:
                    if isinstance(local_result, list):
                        local_result = [
                            r for r in local_result if r not in parent_result
                        ]
                    else:
                        local_result = [local_result]
                    if isinstance(parent_result, list):
                        parent_result.extend(local_result)
                    else:
                        parent_result = [parent_result, *local_result]
            if isinstance(parent_result, list):
                return TensorDict.lazy_stack(
                    [self._from_tensordict(r) for r in parent_result]
                )
            return self._from_tensordict(parent_result)

    @property
    def num_children(self) -> int:
        """Number of children of this node.

        Equates to the number of elements in the ``self.subtree`` stack.
        """
        return len(self.subtree) if self.subtree is not None else 0

    @property
    def is_terminal(self) -> bool | torch.Tensor:
        """Returns True if the tree has no children nodes."""
        if self.rollout is not None:
            return self.rollout[..., -1]["next", "done"].squeeze(-1)
        # If there is no rollout, there is no preceding data - either this is a root or it's a floating node.
        # In either case, we assume that the node is not terminal.
        return False

    def fully_expanded(self, env: EnvBase) -> bool:
        """Returns True if the number of children is equal to the environment cardinality."""
        cardinality = env.cardinality(self.node_data)
        num_actions = self.num_children
        return cardinality == num_actions

    def get_vertex_by_id(self, id: int) -> Tree:
        """Goes through the tree and returns the node corresponding the given id."""
        q = deque()
        q.append(self)
        while len(q):
            tree = q.popleft()
            if tree.node_id == id:
                return tree
            if tree.subtree is not None:
                q.extend(tree.subtree.unbind(0))
        raise ValueError(f"Node with id {id} not found.")

    def get_vertex_by_hash(self, hash: int) -> Tree:
        """Goes through the tree and returns the node corresponding the given hash."""
        q = deque()
        q.append(self)
        while len(q):
            tree = q.popleft()
            if tree.hash == hash:
                return tree
            if tree.subtree is not None:
                q.extend(tree.subtree.unbind(0))
        raise ValueError(f"Node with hash {hash} not found.")

    def __contains__(self, other: Tree) -> bool:
        hash = other.hash
        for vertex in self.vertices().values():
            if vertex.hash == hash:
                return True
        else:
            return False

    def vertices(
        self, *, key_type: Literal["id", "hash", "path"] = "hash"
    ) -> Dict[int | Tuple[int], Tree]:
        """Returns a map containing the vertices of the Tree.

        Keyword args:
            key_type (Literal["id", "hash", "path"], optional): Specifies the type of key to use for the vertices.

                - "id": Use the vertex ID as the key.
                - "hash": Use a hash of the vertex as the key.
                - "path": Use the path to the vertex as the key. This may lead to a dictionary with a longer length than
                    when ``"id"`` or ``"hash"`` are used as the same node may be part of multiple trajectories.
                    Defaults to ``"hash"``.

                Defaults to an empty string, which may imply a default behavior.

        Returns:
            Dict[int | Tuple[int], Tree]: A dictionary mapping keys to Tree vertices.

        """
        memo = set()
        result = {}
        q = deque()
        cur_path = ()
        q.append((self, cur_path))
        use_hash = key_type == "hash"
        use_id = key_type == "id"
        use_path = key_type == "path"
        while len(q):
            tree, cur_path = q.popleft()
            h = tree.hash
            if h in memo and not use_path:
                continue
            memo.add(h)
            if use_path:
                result[cur_path] = tree
            elif use_id:
                result[tree.node_id] = tree
            elif use_hash:
                result[tree.node_id] = tree
            else:
                raise ValueError(
                    f"key_type must be either 'hash', 'id' or 'path'. Got {key_type}."
                )

            n = int(tree.num_children)
            for i in range(n):
                cur_path_tree = cur_path + (i,)
                q.append((tree.subtree[i], cur_path_tree))
        return result

    def num_vertices(self, *, count_repeat: bool = False) -> int:
        """Returns the number of unique vertices in the Tree.

        Keyword Args:
            count_repeat (bool, optional): Determines whether to count repeated vertices.
                - If ``False``, counts each unique vertex only once.
                - If ``True``, counts vertices multiple times if they appear in different paths.
                Defaults to ``False``.

        Returns:
            int: The number of unique vertices in the Tree.

        """
        return len(
            {
                v.node_id
                for v in self.vertices(
                    key_type="hash" if not count_repeat else "path"
                ).values()
            }
        )

    def edges(self) -> List[Tuple[int, int]]:
        """Retrieves a list of edges in the tree.

        Each edge is represented as a tuple of two node IDs: the parent node ID and the child node ID.
        The tree is traversed using Breadth-First Search (BFS) to ensure all edges are visited.

        Returns:
            A list of tuples, where each tuple contains a parent node ID and a child node ID.
        """
        result = []
        q = deque()
        parent = self.node_id
        q.append((self, parent))
        while len(q):
            tree, parent = q.popleft()
            n = int(tree.num_children)
            for i in range(n):
                node = tree.subtree[i]
                node_id = node.node_id
                result.append((parent, node_id))
                q.append((node, node_id))
        return result

    def valid_paths(self):
        """Generates all valid paths in the tree.

        A valid path is a sequence of child indices that starts at the root node and ends at a leaf node.
        Each path is represented as a tuple of integers, where each integer corresponds to the index of a child node.

        Yields:
            tuple: A valid path in the tree.
        """
        # Initialize a queue with the current tree node and an empty path
        q = deque()
        cur_path = ()
        q.append((self, cur_path))
        # Perform BFS traversal of the tree
        while len(q):
            # Dequeue the next tree node and its current path
            tree, cur_path = q.popleft()
            # Get the number of child nodes
            n = int(tree.num_children)
            # If this is a leaf node, yield the current path
            if not n:
                yield cur_path
            # Iterate over the child nodes
            for i in range(n):
                cur_path_tree = cur_path + (i,)
                q.append((tree.subtree[i], cur_path_tree))

    def max_length(self):
        """Returns the maximum length of all valid paths in the tree.

        The length of a path is defined as the number of nodes in the path.
        If the tree is empty, returns 0.

        Returns:
            int: The maximum length of all valid paths in the tree.

        """
        lengths = tuple(len(path) for path in self.valid_paths())
        if len(lengths) == 0:
            return 0
        elif len(lengths) == 1:
            return lengths[0]
        return max(*lengths)

    def rollout_from_path(self, path: Tuple[int]) -> TensorDictBase | None:
        """Retrieves the rollout data along a given path in the tree.

        The rollout data is concatenated along the last dimension (dim=-1) for each node in the path.
        If no rollout data is found along the path, returns ``None``.

        Args:
            path: A tuple of integers representing the path in the tree.

        Returns:
            The concatenated rollout data along the path, or None if no data is found.

        """
        r = self.rollout
        tree = self
        rollouts = []
        if r is not None:
            rollouts.append(r)
        for i in path:
            tree = tree.subtree[i]
            r = tree.rollout
            if r is not None:
                rollouts.append(r)
        if rollouts:
            return torch.cat(rollouts, dim=-1)

    @staticmethod
    def _label(info: List[str], tree: "Tree", root=False):
        labels = []
        for key in info:
            if key == "hash":
                hash = tree.hash
                if hash is not None:
                    hash = hash.item()
                v = f"hash={hash}"
            elif root:
                v = f"{key}=None"
            else:
                v = f"{key}={tree.rollout[key].mean().item()}"

            labels.append(v)
        return ", ".join(labels)

    def plot(
        self: Tree,
        backend: str = "plotly",
        figure: str = "tree",
        info: List[str] = None,
        make_labels: Callable[[Any, ...], Any] | None = None,
    ):
        """Plots a visualization of the tree using the specified backend and figure type.

        Args:
            backend: The plotting backend to use. Currently only supports 'plotly'.
            figure: The type of figure to plot. Can be either 'tree' or 'box'.
            info: A list of additional information to include in the plot (not currently used).
            make_labels: An optional function to generate custom labels for the plot.

        Raises:
            NotImplementedError: If an unsupported backend or figure type is specified.
        """
        if backend == "plotly":
            if figure == "box":
                _plot_plotly_box(self)
                return
            elif figure == "tree":
                _plot_plotly_tree(self, make_labels=make_labels)
                return
            else:
                pass
        raise NotImplementedError(
            f"Unknown plotting backend {backend} with figure {figure}."
        )


class MCTSForest:
    """A collection of MCTS trees.

    .. warning:: This class is currently under active development. Expect frequent API changes.

    The class is aimed at storing rollouts in a storage, and produce trees based on a given root
    in that dataset.

    Keyword Args:
        data_map (TensorDictMap, optional): the storage to use to store the data
            (observation, reward, states etc). If not provided, it is lazily
            initialized using :meth:`~torchrl.data.map.tdstorage.TensorDictMap.from_tensordict_pair`
            using the list of :attr:`observation_keys` and :attr:`action_keys` as ``in_keys``.
        node_map (TensorDictMap, optional): a map from the observation space to the index space.
            Internally, the node map is used to gather all possible branches coming out of
            a given node. For example, if an observation has two associated actions and outcomes
            in the data map, then the :attr:`node_map` will return a data structure containing the
            two indices in the :attr:`data_map` that correspond to these two outcomes.
            If not provided, it is lazily initialized using
            :meth:`~torchrl.data.map.tdstorage.TensorDictMap.from_tensordict_pair` using the list of
            :attr:`observation_keys` as ``in_keys`` and the :class:`~torchrl.data.QueryModule` as
            ``out_keys``.
        max_size (int, optional): the size of the maps.
            If not provided, defaults to ``data_map.max_size`` if this can be found, then
            ``node_map.max_size``. If none of these are provided, defaults to `1000`.
        done_keys (list of NestedKey, optional): the done keys of the environment. If not provided,
            defaults to ``("done", "terminated", "truncated")``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        action_keys (list of NestedKey, optional): the action keys of the environment. If not provided,
            defaults to ``("action",)``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        reward_keys (list of NestedKey, optional): the reward keys of the environment. If not provided,
            defaults to ``("reward",)``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        observation_keys (list of NestedKey, optional): the observation keys of the environment. If not provided,
            defaults to ``("observation",)``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        excluded_keys (list of NestedKey, optional): a list of keys to exclude from the data storage.
        consolidated (bool, optional): if ``True``, the data_map storage will be consolidated on disk.
            Defaults to ``False``.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> import torch
        >>> from tensordict import TensorDict, LazyStackedTensorDict
        >>> from torchrl.data import TensorDictMap, ListStorage
        >>> from torchrl.data.map.tree import MCTSForest
        >>>
        >>> from torchrl.envs import PendulumEnv, CatTensors, UnsqueezeTransform, StepCounter
        >>> # Create the MCTS Forest
        >>> forest = MCTSForest()
        >>> # Create an environment. We're using a stateless env to be able to query it at any given state (like an oracle)
        >>> env = PendulumEnv()
        >>> obs_keys = list(env.observation_spec.keys(True, True))
        >>> state_keys = set(env.full_state_spec.keys(True, True)) - set(obs_keys)
        >>> # Appending transforms to get an "observation" key that concatenates the observations together
        >>> env = env.append_transform(
        ...     UnsqueezeTransform(
        ...         in_keys=obs_keys,
        ...         out_keys=[("unsqueeze", key) for key in obs_keys],
        ...         dim=-1
        ...     )
        ... )
        >>> env = env.append_transform(
        ...     CatTensors([("unsqueeze", key) for key in obs_keys], "observation")
        ... )
        >>> env = env.append_transform(StepCounter())
        >>> env.set_seed(0)
        >>> # Get a reset state, then make a rollout out of it
        >>> reset_state = env.reset()
        >>> rollout0 = env.rollout(6, auto_reset=False, tensordict=reset_state.clone())
        >>> # Append the rollout to the forest. We're removing the state entries for clarity
        >>> rollout0 = rollout0.copy()
        >>> rollout0.exclude(*state_keys, inplace=True).get("next").exclude(*state_keys, inplace=True)
        >>> forest.extend(rollout0)
        >>> # The forest should have 6 elements (the length of the rollout)
        >>> assert len(forest) == 6
        >>> # Let's make another rollout from the same reset state
        >>> rollout1 = env.rollout(6, auto_reset=False, tensordict=reset_state.clone())
        >>> rollout1.exclude(*state_keys, inplace=True).get("next").exclude(*state_keys, inplace=True)
        >>> forest.extend(rollout1)
        >>> assert len(forest) == 12
        >>> # Let's make another final rollout from an intermediate step in the second rollout
        >>> rollout1b = env.rollout(6, auto_reset=False, tensordict=rollout1[3].exclude("next"))
        >>> rollout1b.exclude(*state_keys, inplace=True)
        >>> rollout1b.get("next").exclude(*state_keys, inplace=True)
        >>> forest.extend(rollout1b)
        >>> assert len(forest) == 18
        >>> # Since we have 2 rollouts starting at the same state, our tree should have two
        >>> #  branches if we produce it from the reset entry. Take the state, and call `get_tree`:
        >>> r = rollout0[0]
        >>> # Let's get the compact tree that follows the initial reset. A compact tree is
        >>> #  a tree where nodes that have a single child are collapsed.
        >>> tree = forest.get_tree(r)
        >>> print(tree.max_length())
        2
        >>> print(list(tree.valid_paths()))
        [(0,), (1, 0), (1, 1)]
        >>> from tensordict import assert_close
        >>> # We can manually rebuild the tree
        >>> assert_close(
        ...     rollout1,
        ...     torch.cat([tree.subtree[1].rollout, tree.subtree[1].subtree[0].rollout]),
        ...     intersection=True,
        ... )
        True
        >>> # Or we can rebuild it using the dedicated method
        >>> assert_close(
        ...     rollout1,
        ...     tree.rollout_from_path((1, 0)),
        ...     intersection=True,
        ... )
        True
        >>> tree.plot()
        >>> tree = forest.get_tree(r, compact=False)
        >>> print(tree.max_length())
        9
        >>> print(list(tree.valid_paths()))
        [(0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 0, 0, 0)]
        >>> assert_close(
        ...     rollout1,
        ...     tree.rollout_from_path((1, 0, 0, 0, 0, 0)),
        ...     intersection=True,
        ... )
        True
    """

    def __init__(
        self,
        *,
        data_map: TensorDictMap | None = None,
        node_map: TensorDictMap | None = None,
        max_size: int | None = None,
        done_keys: List[NestedKey] | None = None,
        reward_keys: List[NestedKey] = None,
        observation_keys: List[NestedKey] = None,
        action_keys: List[NestedKey] = None,
        excluded_keys: List[NestedKey] = None,
        consolidated: bool | None = None,
    ):

        self.data_map = data_map

        self.node_map = node_map

        if max_size is None:
            if data_map is not None:
                max_size = data_map.max_size
                if max_size != getattr(node_map, "max_size", max_size):
                    raise ValueError(
                        f"Conflicting max_size: got data_map.max_size={data_map.max_size} and node_map.max_size={node_map.max_size}."
                    )
            elif node_map is not None:
                max_size = node_map.max_size
            else:
                max_size = None
        elif data_map is not None and max_size != getattr(
            data_map, "max_size", max_size
        ):
            raise ValueError(
                f"Conflicting max_size: got data_map.max_size={data_map.max_size} and max_size={max_size}."
            )
        elif node_map is not None and max_size != getattr(
            node_map, "max_size", max_size
        ):
            raise ValueError(
                f"Conflicting max_size: got node_map.max_size={node_map.max_size} and max_size={max_size}."
            )
        self.max_size = max_size

        self.done_keys = done_keys
        self.action_keys = action_keys
        self.reward_keys = reward_keys
        self.observation_keys = observation_keys
        self.excluded_keys = excluded_keys
        self.consolidated = consolidated

    @property
    def done_keys(self) -> List[NestedKey]:
        """Done Keys.

        Returns the keys used to indicate that an episode has ended.
        The default done keys are "done", "terminated", and "truncated". These keys can be
        used in the environment's output to signal the end of an episode.

        Returns:
            A list of strings representing the done keys.

        """
        done_keys = getattr(self, "_done_keys", None)
        if done_keys is None:
            self._done_keys = done_keys = ["done", "terminated", "truncated"]
        return done_keys

    @done_keys.setter
    def done_keys(self, value):
        self._done_keys = _make_list_of_nestedkeys(value, "done_keys")

    @property
    def reward_keys(self) -> List[NestedKey]:
        """Reward Keys.

        Returns the keys used to retrieve rewards from the environment's output.
        The default reward key is "reward".

        Returns:
            A list of strings or tuples representing the reward keys.

        """
        reward_keys = getattr(self, "_reward_keys", None)
        if reward_keys is None:
            self._reward_keys = reward_keys = ["reward"]
        return reward_keys

    @reward_keys.setter
    def reward_keys(self, value):
        self._reward_keys = _make_list_of_nestedkeys(value, "reward_keys")

    @property
    def action_keys(self) -> List[NestedKey]:
        """Action Keys.

        Returns the keys used to retrieve actions from the environment's input.
        The default action key is "action".

        Returns:
            A list of strings or tuples representing the action keys.

        """
        action_keys = getattr(self, "_action_keys", None)
        if action_keys is None:
            self._action_keys = action_keys = ["action"]
        return action_keys

    @action_keys.setter
    def action_keys(self, value):
        self._action_keys = _make_list_of_nestedkeys(value, "action_keys")

    @property
    def observation_keys(self) -> List[NestedKey]:
        """Observation Keys.

        Returns the keys used to retrieve observations from the environment's output.
        The default observation key is "observation".

        Returns:
            A list of strings or tuples representing the observation keys.
        """
        observation_keys = getattr(self, "_observation_keys", None)
        if observation_keys is None:
            self._observation_keys = observation_keys = ["observation"]
        return observation_keys

    @observation_keys.setter
    def observation_keys(self, value):
        self._observation_keys = _make_list_of_nestedkeys(value, "observation_keys")

    @property
    def excluded_keys(self) -> List[NestedKey] | None:
        return self._excluded_keys

    @excluded_keys.setter
    def excluded_keys(self, value):
        self._excluded_keys = _make_list_of_nestedkeys(value, "excluded_keys")

    def get_keys_from_env(self, env: EnvBase):
        """Writes missing done, action and reward keys to the Forest given an environment.

        Existing keys are not overwritten.
        """
        if getattr(self, "_reward_keys", None) is None:
            self.reward_keys = env.reward_keys
        if getattr(self, "_done_keys", None) is None:
            self.done_keys = env.done_keys
        if getattr(self, "_action_keys", None) is None:
            self.action_keys = env.action_keys
        if getattr(self, "_observation_keys", None) is None:
            self.observation_keys = env.observation_keys

    @classmethod
    def _write_fn_stack(cls, new, old=None):
        # This function updates the old values by adding the new ones
        # if and only if the new ones are not there.
        # If the old value is not provided, we assume there are none and the
        # `new` is just prepared.
        # This involves unsqueezing the last dim (since we'll be stacking tensors
        # and calling unique).
        # The update involves calling cat along the last dim + unique
        # which will keep only the new values that were unknown to
        # the storage.
        # We use this method to track all the indices that are associated with
        # an observation. Every time a new index is obtained, it is stacked alongside
        # the others.
        if old is None:
            # we unsqueeze the values to stack them along dim -1
            result = new.apply(lambda x: x.unsqueeze(-1), filter_empty=False)
            result.set(
                "count", torch.ones(result.shape, dtype=torch.int, device=result.device)
            )
        else:

            def cat(name, x, y):
                if name == "count":
                    return x
                if y.ndim < x.ndim:
                    y = y.unsqueeze(-1)
                result = torch.cat([x, y], -1)
                # Breaks on mps
                if result.device.type == "mps":
                    result = result.cpu()
                    result = result.unique(dim=-1, sorted=False)
                    result = result.to("mps")
                else:
                    result = result.unique(dim=-1, sorted=False)
                return result

            result = old.named_apply(cat, new, default=None)
            result.set_("count", old.get("count") + 1)
        return result

    def _make_data_map(self, source, dest):
        try:
            kwargs = {}
            if self.max_size is not None:
                kwargs["max_size"] = self.max_size
            self.data_map = TensorDictMap.from_tensordict_pair(
                source,
                dest,
                in_keys=[*self.observation_keys, *self.action_keys],
                consolidated=self.consolidated,
                **kwargs,
            )
            if self.max_size is None:
                self.max_size = self.data_map.max_size
        except KeyError as err:
            raise KeyError(
                "A KeyError occurred during data map creation. This could be due to the wrong setting of a key in the MCTSForest constructor. Scroll up for more info."
            ) from err

    def _make_node_map(self, source, dest):
        kwargs = {}
        if self.max_size is not None:
            kwargs["max_size"] = self.max_size
        self.node_map = TensorDictMap.from_tensordict_pair(
            source,
            dest,
            in_keys=[*self.observation_keys],
            out_keys=[
                *self.data_map.query_module.out_keys,  # hash and index
                # *self.action_keys,
                # *[("next", rk) for rk in self.reward_keys],
                "count",
            ],
            storage_constructor=ListStorage,
            collate_fn=TensorDict.lazy_stack,
            write_fn=self._write_fn_stack,
            **kwargs,
        )
        if self.max_size is None:
            self.max_size = self.data_map.max_size

    def extend(self, rollout, *, return_node: bool = False):
        source, dest = (
            rollout.exclude("next").copy(),
            rollout.select("next", *self.action_keys).copy(),
        )
        if self.excluded_keys is not None:
            dest = dest.exclude(*self.excluded_keys, inplace=True)
            dest.get("next").exclude(*self.excluded_keys, inplace=True)

        if self.data_map is None:
            self._make_data_map(source, dest)

        # We need to set the action somewhere to keep track of what action lead to what child
        # # Set the action in the 'next'
        # dest[1:] = source[:-1].exclude(*self.done_keys)

        # Add ('observation', 'action') -> ('next, observation')
        self.data_map[source] = dest
        value = source
        if self.node_map is None:
            self._make_node_map(source, dest)
        # map ('observation',) -> ('indices',)
        self.node_map[source] = TensorDict.lazy_stack(value.unbind(0))
        if return_node:
            return self.get_tree(rollout)

    def add(self, step, *, return_node: bool = False):
        source, dest = (
            step.exclude("next").copy(),
            step.select("next", *self.action_keys).copy(),
        )

        if self.data_map is None:
            self._make_data_map(source, dest)

        # We need to set the action somewhere to keep track of what action lead to what child
        # # Set the action in the 'next'
        # dest[1:] = source[:-1].exclude(*self.done_keys)

        # Add ('observation', 'action') -> ('next, observation')
        self.data_map[source] = dest
        value = source
        if self.node_map is None:
            self._make_node_map(source, dest)
        # map ('observation',) -> ('indices',)
        self.node_map[source] = value
        if return_node:
            return self.get_tree(step)

    def get_child(self, root: TensorDictBase) -> TensorDictBase:
        return self.data_map[root]

    def _make_local_tree(
        self,
        root: TensorDictBase,
        index: torch.Tensor | None = None,
        compact: bool = True,
    ) -> Tuple[Tree, torch.Tensor | None, torch.Tensor | None]:
        root = root.select(*self.node_map.in_keys)
        node_meta = None
        if root in self.node_map:
            node_meta = self.node_map[root]
        if index is None:
            node_meta = self.node_map[root]
            index = node_meta["_index"]
        elif index is not None:
            pass
        else:
            return None
        steps = []
        while index.numel() <= 1:
            index = index.squeeze()
            d = self.data_map.storage[index]

            # Rebuild rollout step
            steps.append(merge_tensordicts(d, root, callback_exist=lambda *x: None))
            d = d["next"]
            if d in self.node_map:
                root = d.select(*self.node_map.in_keys)
                node_meta = self.node_map[root]
                index = node_meta["_index"]
                if not compact:
                    break
            else:
                # If the root is provided and not gathered from the storage, it could be that its
                # device doesn't match the data_map storage device.
                root = steps[-1]["next"].select(*self.node_map.in_keys)
                device = getattr(self.data_map.storage, "device", None)
                if root.device != device:
                    if device is not None:
                        root = root.to(self.data_map.storage.device)
                    else:
                        root.clear_device_()
                index = None
                break
        rollout = None
        if steps:
            rollout = torch.stack(steps, -1)
        # Will be populated later
        hash = node_meta["_hash"]
        return (
            Tree(
                rollout=rollout,
                count=torch.zeros((), dtype=torch.int32),
                wins=torch.zeros(()),
                node_data=root,
                index=index,
                hash=None,
                # We do this to avoid raising an exception as rollout and subtree must be provided together
                subtree=None,
            ),
            index,
            hash,
        )

    # The recursive implementation is slower and less compatible with compile
    # def _make_tree(self, root: TensorDictBase, index: torch.Tensor|None=None)->Tree:
    #     tree, indices = self._make_local_tree(root, index=index)
    #     subtrees = []
    #     if indices is not None:
    #         for i in indices:
    #             subtree = self._make_tree(tree.node, index=i)
    #             subtrees.append(subtree)
    #         subtrees = TensorDict.lazy_stack(subtrees)
    #         tree.subtree = subtrees
    #     return tree
    def _make_tree_iter(
        self, root, index=None, max_depth: int | None = None, compact: bool = True
    ):
        q = deque()
        memo = {}
        tree, indices, hash = self._make_local_tree(root, index=index, compact=compact)
        tree.node_id = 0

        result = tree
        depth = 0
        counter = 1
        if indices is not None:
            q.append((tree, indices, hash, depth))

        while len(q):
            tree, indices, hash, depth = q.popleft()
            extend = max_depth is None or depth < max_depth
            subtrees = []
            for i, h in zip(indices, hash):
                # TODO: remove the .item()
                h = h.item()
                subtree, subtree_indices, subtree_hash = memo.get(h, (None,) * 3)
                if subtree is None:
                    subtree, subtree_indices, subtree_hash = self._make_local_tree(
                        tree.node_data,
                        index=i,
                        compact=compact,
                    )
                    subtree.node_id = counter
                    counter += 1
                    subtree.hash = h
                    memo[h] = (subtree, subtree_indices, subtree_hash)
                else:
                    # We just need to save the two (or more) rollouts
                    subtree_bis, _, _ = self._make_local_tree(
                        tree.node_data,
                        index=i,
                        compact=compact,
                    )
                    if subtree.rollout.ndim == subtree_bis.rollout.ndim:
                        subtree.rollout = TensorDict.stack(
                            [subtree.rollout, subtree_bis.rollout]
                        )
                    else:
                        subtree.rollout = TensorDict.stack(
                            [*subtree.rollout, subtree_bis.rollout]
                        )

                subtrees.append(subtree)
                if extend and subtree_indices is not None:
                    q.append((subtree, subtree_indices, subtree_hash, depth + 1))
            subtrees = TensorDict.lazy_stack(subtrees)
            tree.subtree = subtrees

        return result

    def get_tree(
        self,
        root,
        *,
        max_depth: int | None = None,
        compact: bool = True,
    ) -> Tree:
        return self._make_tree_iter(root=root, max_depth=max_depth, compact=compact)

    @classmethod
    def valid_paths(cls, tree: Tree):
        yield from tree.valid_paths()

    def __len__(self):
        return len(self.data_map)


def _make_list_of_nestedkeys(obj: Any, attr: str) -> List[NestedKey]:
    if obj is None:
        return obj
    if isinstance(obj, (str, tuple)):
        return [obj]
    if not isinstance(obj, list):
        raise ValueError(
            f"{attr} must be a list of NestedKeys or a NestedKey, got {obj}."
        )
    return [unravel_key(key) for key in obj]
