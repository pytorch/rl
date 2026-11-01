# SubgoalTriplet

*class*torchrl.data.SubgoalTriplet(***, *goal_discount: float | None = 0.99*, *policy_goal_discount: float | None = None*, *done_key: [NestedKey](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NestedKey.html#tensordict.NestedKey) = ('next', 'done')*)[[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#SubgoalTriplet)

Expands anchors into in-trajectory `(state, subgoal, goal, policy goal)` records.

Each anchor `i` expands into four records of its own episode, in this
order:

- the anchor `i` itself;
- a subgoal `k` drawn uniformly in `[i, j)`;
- a goal `j > i`;
- a policy goal `g > i` drawn independently of `j`.

This is the sampling scheme of Transitive RL (Park et al., 2026), whose
divide-and-conquer critic update regresses `Q(s_i, a_i, s_j)` toward
`Q(s_i, a_i, s_k) Q(s_k, a_k, s_j)`. Restricting `k` to states the
trajectory actually visited between `i` and `j` is what keeps the
update from exploiting arbitrary subgoals.

If the buffer samples `B` anchors, the batch holds `4 * B` records.
`batch.unflatten(0, (B, 4))` recovers one row per anchor. The unit
reports a per-record `"step_offset"` info entry holding the distance in
steps between each record and its anchor (`0` for the anchor), so that
losses can read the `k - i` and `j - k` horizons from the data. An
anchor sampled on the last record of its episode is moved one step back,
since no goal lies after it, which doubles the sampling weight of each
episode's second-to-last record; `"anchor_index"` keeps the sampled index,
so prioritized replay updates the priority the sampler drew.
`"validity_mask"` is `False` for the four records of an anchor whose
episode has a single record.

This unit requires a [`TensorStorage`](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage)
backed by a TensorDict with a one-dimensional batch, since episode
boundaries are read from the stored `done_key` entry. Offsets are drawn
from the buffer's generator when it has one on the anchors' device, so
seeding the buffer makes its batches reproducible.

Reference: Park, Oberai, Atreya, Levine. "Transitive RL: Value Learning
via Divide and Conquer." ICLR 2026. [https://arxiv.org/abs/2510.22512](https://arxiv.org/abs/2510.22512)

See also [`SubgoalTripletConfig`](torchrl.trainers.algorithms.configs.data.SubgoalTripletConfig.html#torchrl.trainers.algorithms.configs.data.SubgoalTripletConfig)
and [`TransitiveQLoss`](torchrl.objectives.TransitiveQLoss.html#torchrl.objectives.TransitiveQLoss).

Keyword Arguments:

- **goal_discount** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**None**,**optional*) - distribution of the goal
offset `j - i`. A float `gamma` in `(0, 1)` draws the offset
from a geometric distribution with success probability
`1 - gamma`, truncated at the episode end. `None` draws `j`
uniformly among the records after `i`. Defaults to `0.99`.
- **policy_goal_discount** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**None**,**optional*) - same as
`goal_discount` for the policy goal. Defaults to `None`
(uniform).
- **done_key** (*NestedKey**,**optional*) - the key of the end-of-episode flag.
Defaults to `("next", "done")`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
>>> from torchrl.data.replay_buffers import SubgoalTriplet
>>> done = torch.zeros(20, 1, dtype=torch.bool)
>>> done[9] = done[19] = True
>>> data = TensorDict(
... {"step": torch.arange(20), ("next", "done"): done}, batch_size=[20]
... )
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(20),
... batch_size=8,
... sample_unit=SubgoalTriplet(goal_discount=0.9),
... )
>>> _ = rb.extend(data)
>>> batch = rb.sample().unflatten(0, (8, 4))
>>> batch.shape
torch.Size([8, 4])
>>> state, subgoal, goal = batch["step"].unbind(-1)[:3]
>>> bool(((state <= subgoal) & (subgoal < goal)).all())
True
>>> bool((batch["step_offset"][:, 2] == goal - state).all())
True
```

expand(*index: [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple*, *info: dict[str, Any]*, *storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*) → tuple[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple, dict[str, Any]][[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#SubgoalTriplet.expand)

Expands anchor indices into the final record indices of the batch.

Parameters:

- **index** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**tuple**of*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the anchor indices
selected by the sampler.
- **info** (*dict*) - the sampler's info dictionary.
- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the storage the batch will be read from.

Returns:

A tuple `(index, info)` with the expanded indices and the
(possibly augmented) info dictionary.