# Sequence

*class*torchrl.data.Sequence(*length: int*, *episode_boundary: Literal['pad', 'stop', 'include_reset'] = 'pad'*, *done_key: NestedKey | None = ('next', 'done')*)[[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#Sequence)

Expands anchors into a fixed-length sequence of records.

This unit requires a [`TensorStorage`](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage)
backed by a TensorDict (e.g. `LazyTensorStorage`
filled with TensorDict data), since episode boundaries are read from the
stored `done_key` entry.

Parameters:

- **length** (*int*) - the length of the sequences.
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
Defaults to `("next", "done")`.

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
['index', 'sequence_id', 'step_in_sequence', 'validity_mask']
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