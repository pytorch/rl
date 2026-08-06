# Sequence

*class*torchrl.data.Sequence(*length: int*, *episode_boundary: Literal['pad', 'stop', 'include_reset'] = 'pad'*, *done_key: NestedKey | None = ('next', 'done')*, *burn_in: int = 0*, *bootstrap: int = 0*, *dilation: int = 1*)[[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#Sequence)

Expands anchors into a window of records around each anchor.

Each anchor expands into `burn_in + length + bootstrap` records:
`burn_in` records preceding the anchor, the learning region of
`length` records starting at the anchor, then `bootstrap` records
following it. `dilation` spaces the records of the window uniformly.

This is useful when a learner needs temporal context around the records
that contribute to its loss. For example, a recurrent Q-learning learner
can replay the burn-in prefix to reconstruct its hidden state, compute
losses only over the learning region, and use the bootstrap suffix as
future context for a multi-step target. The unit only selects stored
records and reports masks: it does not run the recurrent model or compute
bootstrap targets.

The configuration is fixed when the unit is constructed. If the replay
buffer samples `B` anchors, the returned flat batch contains
`B * (burn_in + length + bootstrap)` records. The corresponding storage
span is `1 + dilation * (burn_in + length + bootstrap - 1)` records.

This unit requires a [`TensorStorage`](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage)
backed by a TensorDict (e.g. `LazyTensorStorage`
filled with TensorDict data), since episode boundaries are read from the
stored `done_key` entry.

Parameters:

- **length** (*int*) - the length of the learning region of the sequences.
- **episode_boundary** (*str**,**optional*) -

boundary policy. One of:

- `"pad"`: repeat the last valid state if a boundary is reached,
marking padded steps as invalid in the `"validity_mask"` info
entry.
- `"stop"`: shift the anchor backward so the sequence ends
exactly at the boundary, falling back to pad if the episode is
shorter than `length`.
- `"include_reset"`: cross episode boundaries. The write seam
(the boundary between the newest and the oldest record of the
ring buffer) and unwritten slots are never crossed: records
beyond it are clamped and marked invalid.

Defaults to `"pad"`.
- **done_key** (*NestedKey**,**optional*) - the key for the end-of-episode flag.
If `None`, the written storage span is treated as one trajectory,
bounded only by the replay buffer's write seam. Defaults to
`("next", "done")`.
- **burn_in** (*int**,**optional*) - number of records preceding the anchor,
marked False in the `"learning_mask"` info entry. Burn-in never
shifts the anchor: entries before the anchor's episode start (or
before the oldest written record) are invalid and clamp to that
boundary. A recurrent learner can process these records to
reconstruct its hidden state while excluding them from the loss.
Defaults to 0.
- **bootstrap** (*int**,**optional*) - number of records following the learning
region, marked False in `"learning_mask"` and subject to the
`episode_boundary` policy at episode ends. These records provide
future context to a target estimator; this unit does not compute a
bootstrap value. Defaults to 0.
- **dilation** (*int**,**optional*) - distance in storage records between
consecutive records of the returned window. For example,
`dilation=2` selects every other stored record. Dilation does not
aggregate skipped transitions and does not control the spacing or
overlap between independently sampled windows. Defaults to 1.

After expansion, `info["index"]` (and the `"index"` entry of
TensorDict samples) holds the expanded per-record storage indices of the
window records, not the anchors. The unit additionally reports a
per-record `"anchor_index"` info entry holding the storage index of
each record's sampled anchor, so priorities of sampled sequences can be
updated per anchor through `update_priority`.
`TensorDictReplayBuffer.update_tensordict_priority` uses it
automatically: per-record priorities are reduced (max over the valid
records of each window) and written to the anchors only, so padded or
bootstrap records never pollute the priorities of unrelated anchors.

Note

`"anchor_index"` always reports the anchor the sampler drew.
With `episode_boundary="stop"` the window may be shifted backward,
and with `dilation > 1` the shifted window is laid out on the
dilation grid of the shifted anchor: the reported (pre-shift) anchor
is then not necessarily one of the window's records.

See also

`SequenceConfig`
for the Hydra configuration companion.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer, Sequence
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(10),
... batch_size=2,
... sample_unit=Sequence(length=3),
... )
>>> done = torch.zeros(10, 1, dtype=torch.bool)
>>> done[4] = done[9] = True
>>> rb.extend(TensorDict(
... {
... "obs": torch.arange(10, dtype=torch.float32),
... ("next", "done"): done,
... },
... batch_size=[10],
... ))
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> sample, info = rb.sample(return_info=True)
>>> sample["obs"].shape # 2 anchors x 3 records each
torch.Size([6])
>>> sorted(info.keys())
['anchor_index', 'index', 'learning_mask', 'sequence_id', 'step_in_sequence', 'validity_mask']
>>> # A recurrent learner could use one record to warm up its hidden
>>> # state, learn on two records, and keep one future record for its
>>> # target. It runs over the full window and applies the loss only
>>> # where learning_mask and validity_mask are both true.
>>> unit = Sequence(length=2, burn_in=1, bootstrap=1)
>>> index, info = unit.expand(torch.tensor([5]), {}, rb.storage)
>>> index.tolist() # burn-in clamps at the episode start (5)
[5, 5, 6, 7]
>>> info["learning_mask"].tolist()
[False, True, True, False]
>>> info["validity_mask"].tolist()
[False, True, True, True]
>>> # Dilation temporally subsamples records inside the window:
>>> index, _ = Sequence(length=3, dilation=2).expand(
... torch.tensor([0]), {}, rb.storage
... )
>>> index.tolist()
[0, 2, 4]
```

expand(*index: [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple*, *info: dict[str, Any]*, *storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*) → tuple[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple, dict[str, Any]][[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#Sequence.expand)

Expands anchor indices into the final record indices of the batch.

Parameters:

- **index** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**tuple**of*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the anchor indices
selected by the sampler.
- **info** (*dict*) - the sampler's info dictionary.
- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the storage the batch will be read from.

Returns:

A tuple `(index, info)` with the expanded indices and the
(possibly augmented) info dictionary.