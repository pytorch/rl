# find_start_stop_traj

torchrl.data.find_start_stop_traj(***, *trajectory: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *end: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *at_capacity: bool*, *cursor: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | range | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/data/replay_buffers/utils.html#find_start_stop_traj)

Recover trajectory boundaries from trajectory ids or end-of-trajectory flags.

This is the canonical trajectory-boundary recovery routine used by
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) and its subclasses. It
understands the storage layout of TorchRL's replay buffers: circular
(ring-buffer) storages where a trajectory may span the wrap point, and
partially-filled storages where the write cursor acts as an implicit
truncation. See [the trajectory-boundary documentation](../data_layout.html#ref-traj-boundaries) for the full contract.

Keyword Arguments:

- **trajectory** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a tensor of trajectory ids laid
out in storage order, with time along dim `0` and any extra
batch dims after it (shape `[T, *B]`). A boundary is detected
wherever the id changes between two consecutive steps. Exclusive
with `end`.
- **end** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a boolean tensor of end-of-trajectory
flags laid out in storage order (shape `[T, *B]`, time along
dim `0`). `True` marks the last step of a trajectory.
Exclusive with `trajectory`.
- **at_capacity** (*bool*) - whether the storage is full and behaves as a
circular buffer. If `True`, a trajectory that has no end flag
after the last row is assumed to continue at row `0` (it spans
the wrap point). If `False`, the last valid row is always
treated as an end.
- **cursor** (*int**,*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**range**,**optional*) - the index of the last
written row (e.g. `storage._last_cursor`). Only used when
`at_capacity=True`: the row under the cursor is forced to be an
end, since the data that followed it has been overwritten and the
stored trajectory is implicitly truncated there.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - a device on which to run the
boundary computation (the underlying `nonzero()` call can
benefit from an accelerator for large storages). Results are
returned on the device of the input tensor. Defaults to `None`
(compute where the input lives).

Returns:

A `(start, stop, lengths)` tuple where `start` and `stop` are
`[N, 1 + len(B)]` integer tensors holding, for each of the `N`
recovered trajectories, the time index of its first (resp. last,
inclusive) step in column `0` and the batch coordinates in the
remaining columns; `lengths` is a `[N]` tensor of trajectory
lengths. For a trajectory spanning the wrap point of a full circular
storage, `start[i, 0] > stop[i, 0]` and the length accounts for the
wrap.

Examples

```
>>> import torch
>>> from torchrl.data import find_start_stop_traj
>>> # A full circular storage with 10 rows and end flags at rows 2 and 7.
>>> # The write cursor sits at row 4: row 4 is an implicit truncation.
>>> end = torch.zeros(10, dtype=torch.bool)
>>> end[2] = end[7] = True
>>> start, stop, lengths = find_start_stop_traj(end=end, at_capacity=True, cursor=4)
>>> start.squeeze(-1) # the trajectory starting at row 8 wraps around to row 2
tensor([8, 3, 5])
>>> stop.squeeze(-1)
tensor([2, 4, 7])
>>> lengths
tensor([5, 2, 3])
>>> # The same boundaries recovered from trajectory ids
>>> trajectory = torch.tensor([5, 5, 5, 0, 0, 1, 1, 1, 5, 5])
>>> start, stop, lengths = find_start_stop_traj(trajectory=trajectory, at_capacity=True)
>>> start.squeeze(-1), stop.squeeze(-1), lengths
(tensor([8, 3, 5]), tensor([2, 4, 7]), tensor([5, 2, 3]))
```

See also

[`split_trajectories()`](torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) splits a
contiguous *rollout batch* (fresh collector output) into a padded
`[B, T]` layout. This function instead operates on *storage-order*
data and returns indices, leaving the data untouched.