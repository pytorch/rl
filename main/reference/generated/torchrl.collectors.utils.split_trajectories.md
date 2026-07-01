# split_trajectories

torchrl.collectors.utils.split_trajectories(*rollout_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *prefix=None*, *trajectory_key: NestedKey | None = None*, *done_key: NestedKey | None = None*, *as_nested: bool = False*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/collectors/utils.html#split_trajectories)

A util function for trajectory separation.

Takes a tensordict with a key traj_ids that indicates the id of each trajectory.

From there, builds a B x T x ... zero-padded tensordict with B batches on max duration T

Parameters:

**rollout_tensordict** (*TensorDictBase*) - a rollout with adjacent trajectories
along the last dimension.

Keyword Arguments:

- **prefix** (*NestedKey**,**optional*) - the prefix used to read and write meta-data,
such as `"traj_ids"` (the optional integer id of each trajectory)
and the `"mask"` entry indicating which data are valid and which
aren't. Defaults to `"collector"` if the input has a `"collector"`
entry, `()` (no prefix) otherwise.
`prefix` is kept as a legacy feature and will be deprecated eventually.
Prefer `trajectory_key` or `done_key` whenever possible.
- **trajectory_key** (*NestedKey**,**optional*) - the key pointing to the trajectory
ids. Supersedes `done_key` and `prefix`. If not provided, defaults
to `(prefix, "traj_ids")`.
- **done_key** (*NestedKey**,**optional*) - the key pointing to the `"done""` signal,
if the trajectory could not be directly recovered. Defaults to `"done"`.
- **as_nested** (*bool**or*[*torch.layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)*,**optional*) -

whether to return the results as nested
tensors. Defaults to `False`. If a `torch.layout` is provided, it will be used
to construct the nested tensor, otherwise the default layout will be used.

Note

Using `split_trajectories(tensordict, as_nested=True).to_padded_tensor(mask=mask_key)`
should result in the exact same result as `as_nested=False`. Since this is an experimental
feature and relies on nested_tensors, which API may change in the future, we made this
an optional feature. The runtime should be faster with `as_nested=True`.

Note

Providing a layout lets the user control whether the nested tensor is to be used
with `torch.strided` or `torch.jagged` layout. While the former has slightly more
capabilities at the time of writing, the second will be the main focus of the PyTorch team
in the future due to its better compatibility with [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

Returns:

A new tensordict with a leading dimension corresponding to the trajectory.
A `"mask"` boolean entry sharing the `trajectory_key` prefix
and the tensordict shape is also added. It indicated the valid elements of the tensordict,
as well as a `"traj_ids"` entry if `trajectory_key` could not be found.

Note

This function splits whatever the input contains: trajectories
spanning several collector batches stay split across the corresponding
calls. To collect batches made of complete trajectories only, pass
`trajs_per_batch` to the collector instead (see
[Complete trajectory collection with trajs_per_batch](../collectors_replay.html#collectors-replay-trajs)).

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> from torchrl.collectors.utils import split_trajectories
>>> obs = torch.cat([torch.arange(10), torch.arange(5)])
>>> obs_ = torch.cat([torch.arange(1, 11), torch.arange(1, 6)])
>>> done = torch.zeros(15, dtype=torch.bool)
>>> done[9] = True
>>> trajectory_id = torch.cat([torch.zeros(10, dtype=torch.int32),
... torch.ones(5, dtype=torch.int32)])
>>> data = TensorDict({"obs": obs, ("next", "obs"): obs_, ("next", "done"): done, "trajectory": trajectory_id}, batch_size=[15])
>>> data_split = split_trajectories(data, done_key="done")
>>> print(data_split)
TensorDict(
 fields={
 mask: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.bool, is_shared=False),
 obs: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([2, 10]),
 device=None,
 is_shared=False),
 obs: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False),
 traj_ids: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False),
 trajectory: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([2, 10]),
 device=None,
 is_shared=False)
>>> # check that split_trajectories got the trajectories right with the done signal
>>> assert (data_split["traj_ids"] == data_split["trajectory"]).all()
>>> print(data_split["mask"])
tensor([[ True, True, True, True, True, True, True, True, True, True],
 [ True, True, True, True, True, False, False, False, False, False]])
>>> data_split = split_trajectories(data, trajectory_key="trajectory")
>>> print(data_split)
TensorDict(
 fields={
 mask: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.bool, is_shared=False),
 obs: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([2, 10]),
 device=None,
 is_shared=False),
 obs: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False),
 trajectory: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([2, 10]),
 device=None,
 is_shared=False)
```