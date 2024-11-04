# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import weakref
from typing import List

import torch
from tensordict import LazyStackedTensorDict, NestedKey, tensorclass, TensorDict
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.data.map.tdstorage import TensorDictMap
from torchrl.envs.common import EnvBase


@tensorclass
class MCTSNode:
    """An MCTS node.

    The batch-size of a root node is indicative of the batch-size of the tree:
    each indexed element of a ``Node`` corresponds to a separate tree.

    A node is characterized by its data (a tensordict with keys such as ``"observation"``,
    or ``"done"``), a ``children`` field containing all the branches from that node
    (one per action taken), and a ``count`` tensor indicating how many times this node
    has been visited.

    """

    data_content: TensorDict
    children: MCTSChildren | None = None
    count: torch.Tensor | None = None
    forest: MCTSForest | None = None

    def plot(self, backend="plotly", info=None):
        forest = self.forest
        if isinstance(forest, weakref.ref):
            forest = forest()
        return forest.plot(self, backend=backend, info=info)

@tensorclass
class MCTSChildren:
    """The children of a node.

    This class contains data of the same batch-size: the ``index`` and ``hash``
    associated with each ``node``. Therefore, each indexed element of a ``Children``
    corresponds to one child with its associated index. Actions .

    """

    node: MCTSNode
    index: torch.Tensor | None = None
    hash: torch.Tensor | None = None


class MCTSForest:
    """A collection of MCTS trees.

    The class is aimed at storing rollouts in a storage, and produce trees based on a given root
    in that dataset.

    Keyword Args:
        data_map (TensorDictMap, optional): the storage to use to store the data
            (observation, reward, states etc). If not provided, it is lazily
            initialized using :meth:`~torchrl.data.map.tdstorage.TensorDictMap.from_tensordict_pair`.
        node_map (TensorDictMap, optional): TODO
        done_keys (list of NestedKey): the done keys of the environment. If not provided,
            defaults to ``("done", "terminated", "truncated")``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        action_keys (list of NestedKey): the action keys of the environment. If not provided,
            defaults to ``("action",)``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        reward_keys (list of NestedKey): the reward keys of the environment. If not provided,
            defaults to ``("reward",)``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.
        observation_keys (list of NestedKey): the observation keys of the environment. If not provided,
            defaults to ``("observation",)``.
            The :meth:`~.get_keys_from_env` can be used to automatically determine the keys.

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
        >>> rollout0.exclude(*state_keys, inplace=True)
        >>> rollout0.get("next").exclude(*state_keys, inplace=True)
        >>> forest.extend(rollout0)
        >>> # The forest should have 6 elements (the length of the rollout)
        >>> assert len(forest) == 6
        >>> # Let's make another rollout from the same reset state
        >>> rollout1 = env.rollout(6, auto_reset=False, tensordict=reset_state.clone())
        >>> rollout1.exclude(*state_keys, inplace=True)
        >>> rollout1.get("next").exclude(*state_keys, inplace=True)
        >>> forest.extend(rollout1)
        >>> assert len(forest) == 12
        >>> # Since we have 2 rollouts starting at the same state, our tree should have two
        >>> #  branches if we produce it from the reset entry. Take the sate, and call `get_tree`:
        >>> r = rollout0[0]
        >>> tree = forest.get_tree(r)
        >>> tree.shape
        (torch.Size([]), _StringKeys(dict_keys(['data_content', 'children', 'count'])))
         >>> tree.count
         tensor(2, dtype=torch.int32)
        >>> tree.children.shape
        (torch.Size([2]),
         _StringKeys(dict_keys(['node', 'action', 'reward', 'index', 'hash'])))
        >>> # Now, let's run a rollout from an intermediate step from the second rollout
        >>> starttd = rollout1[2]
        >>> rollout2 = env.rollout(6, auto_reset=False, tensordict=starttd)
        >>> rollout2.exclude(*state_keys, inplace=True)
        >>> rollout2.get("next").exclude(*state_keys, inplace=True)
        >>> forest.extend(rollout2)
        >>> assert len(forest) == 18
        >>> # Now our tree is a bit more complex, since from the 3rd level we have 3 different branches
        >>> #  When looking at the third level, we should see that the count for the left branch is 1, whereas the count
        >>> #  for the right branch is 2.
        >>> r = rollout0[0]
        >>> tree = forest.get_tree(r)
        >>> print(tree.children.node.children.node.count, tree.children.node.children.node.shape)
        tensor([[1],
                [2]], dtype=torch.int32) torch.Size([2, 1])
        >>> # This is reflected by the shape of the children: the left has shape [1, 1], the right [1, 2]
        >>> #  and tensordict represents this as shape [2, 1, -1]
        >>> tree.children.node.children.node.children.shape, tree.children.node.children.node.children[0].shape, tree.children.node.children.node.children[1].shape
        (torch.Size([2, 1, -1]), torch.Size([1, 1]), torch.Size([1, 2]))
        >>> tree.plot(info=["step_count", "reward"])

    """

    def __init__(
        self,
        *,
        data_map: TensorDictMap | None = None,
        node_map: TensorDictMap | None = None,
        done_keys: List[NestedKey] | None = None,
        reward_keys: List[NestedKey] = None,
        observation_keys: List[NestedKey] = None,
        action_keys: List[NestedKey] = None,
    ):

        self.data_map = data_map

        self.node_map = node_map

        self.done_keys = done_keys
        self.action_keys = action_keys
        self.reward_keys = reward_keys
        self.observation_keys = observation_keys

    @property
    def done_keys(self):
        done_keys = getattr(self, "_done_keys", None)
        if done_keys is None:
            self._done_keys = done_keys = ("done", "terminated", "truncated")
        return done_keys

    @done_keys.setter
    def done_keys(self, value):
        self._done_keys = value

    @property
    def reward_keys(self):
        reward_keys = getattr(self, "_reward_keys", None)
        if reward_keys is None:
            self._reward_keys = reward_keys = ("reward",)
        return reward_keys

    @reward_keys.setter
    def reward_keys(self, value):
        self._reward_keys = value

    @property
    def action_keys(self):
        action_keys = getattr(self, "_action_keys", None)
        if action_keys is None:
            self._action_keys = action_keys = ("action",)
        return action_keys

    @action_keys.setter
    def action_keys(self, value):
        self._action_keys = value

    @property
    def observation_keys(self):
        observation_keys = getattr(self, "_observation_keys", None)
        if observation_keys is None:
            self._observation_keys = observation_keys = ("observation",)
        return observation_keys

    @observation_keys.setter
    def observation_keys(self, value):
        self._observation_keys = value

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
        if old is None:
            result = new.apply(lambda x: x.unsqueeze(0))
            result.set(
                "count", torch.ones(result.shape, dtype=torch.int, device=result.device)
            )
        else:

            def cat(name, x, y):
                if name == "count":
                    return x
                if y.ndim < x.ndim:
                    y = y.unsqueeze(0)
                result = torch.cat([x, y], 0).unique(dim=0, sorted=False)
                return result

            result = old.named_apply(cat, new, default=None)
            result.set_("count", old.get("count") + 1)
        return result

    def _make_storage(self, source, dest):
        self.data_map = TensorDictMap.from_tensordict_pair(
            source,
            dest,
            in_keys=[*self.observation_keys, *self.action_keys],
        )

    def _make_storage_branches(self, source, dest):
        self.node_map = TensorDictMap.from_tensordict_pair(
            source,
            dest,
            in_keys=[*self.observation_keys],
            out_keys=[
                *self.data_map.query_module.out_keys, # hash and index
                # *self.action_keys,
                # *[("next", rk) for rk in self.reward_keys],
                "count",
            ],
            storage_constructor=ListStorage,
            collate_fn=TensorDict.lazy_stack,
            write_fn=self._write_fn_stack,
        )

    def extend(self, rollout):
        source, dest = rollout.exclude("next").copy(), rollout.select("next", *self.action_keys).copy()

        if self.data_map is None:
            self._make_storage(source, dest)

        # We need to set the action somewhere to keep track of what action lead to what child
        # # Set the action in the 'next'
        # dest[1:] = source[:-1].exclude(*self.done_keys)

        self.data_map[source] = dest
        value = source
        if self.node_map is None:
            self._make_storage_branches(source, dest)
        self.node_map[source] = TensorDict.lazy_stack(value.unbind(0))

    def get_child(self, root):
        return self.data_map[root]

    def get_tree(
        self,
        root,
        *,
        inplace: bool = False,
        recurse: bool = True,
        max_depth: int | None = None,
        as_tensordict: bool = False,
        compact: bool=False,
    ):
        if root.batch_size:
            if compact:
                raise NotImplementedError
            func = self._get_tree_batched
        else:
            if compact:
                func = self._get_tree_single_compact
            else:
                func = self._get_tree_single
        return func(
            root=root,
            inplace=inplace,
            recurse=recurse,
            max_depth=max_depth,
            as_tensordict=as_tensordict,
        )

    def _get_tree_single(
        self,
        root,
        inplace: bool = False,
        recurse: bool = True,
        max_depth: int | None = None,
        as_tensordict: bool = False,
    ):
        if root not in self.node_map:
            if as_tensordict:
                return TensorDict({"data_content": root})
            return MCTSNode(root, forest=weakref.ref(self))

        branches = self.node_map[root]

        index = branches["_index"]
        hash_val = branches["_hash"]
        count = branches["count"]

        children_node = self.data_map.storage[index]
        if not inplace:
            root = root.copy()
        if recurse:
            children_node = children_node.unbind(0)
            children_node = tuple(
                self.get_tree(
                    child,
                    inplace=inplace,
                    max_depth=max_depth - 1 if isinstance(max_depth, int) else None,
                )
                for child in children_node
            )
            if not as_tensordict:
                children_node = LazyStackedTensorDict(
                    *(child._tensordict for child in children_node)
                )
                children_node = MCTSNode.from_tensordict(children_node)
                children_node.forest = weakref.ref(self)
            else:
                children_node = LazyStackedTensorDict(*children_node)

        if not as_tensordict:
            return MCTSNode(
                data_content=root,
                children=MCTSChildren(
                    node=children_node,
                    index=index,
                    hash=hash_val,
                    batch_size=children_node.batch_size,
                ),
                count=count,
                forest=weakref.ref(self),
            )
        return TensorDict(
            {
                "data_content": root,
                "children": TensorDict(
                    {
                        "node": children_node,
                        "index": index,
                        "hash": hash_val,
                    },
                    batch_sizde=children_node.batch_size,
                ),
                "count": count,
            }
        )

    def _get_tree_single_compact(
        self,
        root,
        inplace: bool = False,
        recurse: bool = True,
        max_depth: int | None = None,
        as_tensordict: bool = False,
    ):
        if root not in self.node_map:
            if as_tensordict:
                return TensorDict({"data_content": root})
            return MCTSNode(root, forest=weakref.ref(self))

        stack = []
        child = root.select(*self.observation_keys).copy()
        while True:
            if child not in self.node_map:
                # we already know this doesn't happen during the first iter
                break
            branch = self.node_map[child]
            index = branch["_index"]
            if index.numel() == 1:
                # child contains (action, next) so we can update the previous data with this
                new_child = self.data_map.storage[index[0]]
                child.update(new_child)
                stack.append(child)
                child = new_child.get("next").select(*self.observation_keys)
            else:
                break
        if len(stack):
            root = torch.stack(stack, -1)
        elif not inplace:
            root = child

        hash_val = branch["_hash"]
        count = branch["count"]

        children_node = self.data_map.storage[index].get("next").select(*self.observation_keys)
        if recurse:
            children_node = children_node.unbind(0)
            children_node = tuple(
                self.get_tree(
                    child,
                    inplace=inplace,
                    max_depth=max_depth - 1 if isinstance(max_depth, int) else None,
                    compact=True
                )
                for child in children_node
            )
            if not as_tensordict:
                children_node = LazyStackedTensorDict(
                    *(child._tensordict for child in children_node)
                )
                children_node = MCTSNode.from_tensordict(children_node)
                children_node.forest = weakref.ref(self)
            else:
                children_node = LazyStackedTensorDict(*children_node)

        if not as_tensordict:
            return MCTSNode(
                data_content=root,
                children=MCTSChildren(
                    node=children_node,
                    index=index,
                    hash=hash_val,
                    batch_size=children_node.batch_size,
                ),
                count=count,
                forest=weakref.ref(self),
            )
        return TensorDict(
            {
                "data_content": root,
                "children": TensorDict(
                    {
                        "node": children_node,
                        "index": index,
                        "hash": hash_val,
                    },
                    batch_sizde=children_node.batch_size,
                ),
                "count": count,
            }
        )

    def _get_tree_batched(
        self,
        root,
        inplace: bool = False,
        recurse: bool = True,
        max_depth: int | None = None,
        as_tensordict: bool = False,
    ):
        present = self.node_map.contains(root)
        if not present.any():
            if as_tensordict:
                return TensorDict({"data_content": root}, batch_size=root.batch_size)
            return MCTSNode(root, forest=weakref.ref(self), batch_size=root.batch_size)
        if present.all():
            root_present = root
        else:
            root_present = root[present]
        branches = self.node_map[root_present]
        index = branches.get_nestedtensor("_index", layout=torch.jagged)
        hash_val = branches.get_nestedtensor("_hash", layout=torch.jagged)
        count = branches.get("count")

        children_node = self.data_map.storage[index.values()]
        if not root_present.all():
            children_node = LazyStackedTensorDict(
                *children_node.split(index.offsets().diff().tolist())
            )
            for idx in (~present).nonzero(as_tuple=True)[0].tolist():
                children_node.insert(idx, TensorDict())  # TODO: replace with new_zero

        if not inplace:
            root = root.copy()
        if recurse:
            children_node = children_node.unbind(0)
            children_node = tuple(
                self.get_tree(
                    child,
                    inplace=inplace,
                    max_depth=max_depth - 1 if isinstance(max_depth, int) else None,
                )
                if present[i]
                else child
                for i, child in enumerate(children_node)
            )
        children = TensorDict.lazy_stack(
            [
                TensorDict(
                    {
                        "node": _children_node,
                        "index": _index,
                        "hash": _hash_val,
                    },
                    batch_size=_children_node.batch_size,
                )
                for (_children_node, _action, _index, _hash_val, _reward) in zip(
                    children_node,
                    action.unbind(0),
                    index.unbind(0),
                    hash_val.unbind(0),
                    reward.unbind(0),
                )
            ]
        )
        if not as_tensordict:
            return MCTSNode(
                data_content=root,
                children=MCTSChildren._from_tensordict(children),
                count=count,
                forest=weakref.ref(self),
                batch_size=root.batch_size,
            )
        return TensorDict(
            {
                "data_content": root,
                "children": children,
                "count": count,
            },
            batch_size=root.batch_size,
        )

    def __len__(self):
        return len(self.data_map)

    @staticmethod
    def _label(info, tree, root=False):
        labels = []
        for key in info:
            if key == "hash":
                if not root:
                    v = f"hash={tree.hash.item()}"
                else:
                    v = f"hash={tree.data_content['_hash'].item()}"
            elif root:
                v = f"{key}=None"
            else:
                v = f"{key}={tree.node.data_content[key].item()}"

            labels.append(v)
        return ", ".join(labels)

    def plot(self, tree, backend="plotly", info=None):
        if backend == "plotly":
            import plotly.graph_objects as go
            if info is None:
                info = ["hash", "reward"]

            parents = [""]
            labels = [self._label(info, tree, root=True)]

            _tree = tree

            def extend(tree, parent):
                children = tree.children
                if children is None:
                    return
                for child in children:
                    labels.append(self._label(info, child))
                    parents.append(parent)
                    extend(child.node, labels[-1])

            extend(_tree, labels[-1])
            fig = go.Figure(go.Treemap(labels=labels, parents=parents))
            fig.show()
        else:
            raise NotImplementedError(f"Unkown plotting backend {backend}.")
