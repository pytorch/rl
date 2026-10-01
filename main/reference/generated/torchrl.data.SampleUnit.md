# SampleUnit

*class*torchrl.data.SampleUnit[[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#SampleUnit)

Expands sampled anchors into the records a batch is made of.

Replay sampling combines two orthogonal decisions: which anchors are
selected (the sampler's probability distribution) and what each anchor
expands into (a single transition, a fixed-length sequence, a complete
trajectory). A `SampleUnit` owns the second decision. The buffer calls
`expand()` inside its sampling critical section, after the anchor
sampler ran and before the storage is read or any index bookkeeping
happens, so the indices it returns are the ones the batch is built from
and the ones reported in the sample info.

Contract for implementations:

- `expand` receives the anchor index (a tensor, or a tuple of
coordinate tensors for multidimensional storages), the sampler's info
dictionary and the storage. It returns the expanded index and info,
which may be new objects; it must not mutate the storage.
- Entries of `info` that are aligned with the anchors (for example
priority weights) are the unit's responsibility: a unit that changes
the number of records must expand or reduce those entries so they stay
aligned with the index it returns.
- Metadata describing the expansion (validity masks, learning masks,
per-record anchor provenance) is communicated by adding entries to
`info`; scalar-per-record tensors are surfaced as keys of
TensorDict samples automatically.

See also

[`Transition`](torchrl.data.Transition.html#torchrl.data.Transition), the identity unit reproducing classic
one-anchor-one-transition sampling.

*abstract*expand(*index: [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple*, *info: dict[str, Any]*, *storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*) → tuple[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple, dict[str, Any]][[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#SampleUnit.expand)

Expands anchor indices into the final record indices of the batch.

Parameters:

- **index** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**tuple**of*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the anchor indices
selected by the sampler.
- **info** (*dict*) - the sampler's info dictionary.
- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the storage the batch will be read from.

Returns:

A tuple `(index, info)` with the expanded indices and the
(possibly augmented) info dictionary.