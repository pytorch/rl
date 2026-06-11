# ActionChunkTransform

*class*torchrl.envs.transforms.ActionChunkTransform(*chunk_size: int*, ***, *action_key: NestedKey = 'action'*, *chunk_key: NestedKey = 'action_chunk'*, *pad_key: NestedKey = 'action_is_pad'*, *time_dim: int = -2*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionChunkTransform)

Build fixed-length action chunks from a trajectory window.

Action *chunking* is the defining trait of modern VLA policies (ACT,
OpenVLA-OFT, pi0, SmolVLA): instead of predicting a single action, the
policy predicts a short horizon `H` of future actions. This transform
turns a per-step action tensor `[*B, T, action_dim]` into the
corresponding training target `action_chunk` of shape
`[*B, T, H, action_dim]` - for each time step `t` it gathers the
actions `a[t], a[t+1], ..., a[t+H-1]` - together with a boolean
`action_is_pad` mask `[*B, T, H]` marking the steps that ran past the
end of the window (and were filled by repeating the last available action).

Note

**How to read "many actions in one tensor".** The `H` actions
of a chunk are *predictions* - overlapping, stride-1 training targets
(each dataset step `t` gets its own window `a[t..t+H-1]`, so a
given action appears in up to `H` different chunks) - not a macro
action to be replayed verbatim. This transform is a pure *data*
transform (it builds training targets) and never touches the
environment; how many of the `H` predicted actions actually get
executed per policy call is a separate, execution-time choice:

- [`MultiAction`](torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction) executes every action
in the tensor by stepping the base env once per action with a
single policy call per chunk (one outer step = `H` base steps,
rewards stacked or aggregated);
- [`MultiStepActorWrapper`](torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper)
keeps the env timing unchanged: it caches the predicted actions and
emits one per step, skipping the actor call while the cache lasts
- open-loop by default, receding horizon with
`replan_interval < n_steps`, closed loop with
`replan_interval=1`.

The forward (data) path operates on **time-structured** data: the action
tensor must be shaped `[*B, T, action_dim]` and each row along
`time_dim` must be a single contiguous trajectory window. Chunks are
built independently per row and never cross a row boundary; the downstream
chunked behavior-cloning loss masks the padded steps out using
`action_is_pad`.

Note

A `SliceSampler` returns a *flat* `[B * T, ...]`
batch - reshape it to `[num_slices, slice_len, ...]` before applying
this transform, otherwise chunks would span across trajectory boundaries.
Datasets that store one trajectory window per item (e.g.
[`OpenXExperienceReplay`](torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay)) already yield
time-structured `[batch, T, ...]` samples and can use this transform
directly. When this transform is appended to a replay buffer, the
chunks are built on the `sample` path only; `extend` leaves the
stored (raw, per-step) data untouched.

Parameters:

**chunk_size** (*int*) - the horizon `H` of the action chunk.

Keyword Arguments:

- **action_key** (*NestedKey*) - the per-step action to read.
Defaults to `"action"`.
- **chunk_key** (*NestedKey*) - where to write the action chunk.
Defaults to `"action_chunk"`.
- **pad_key** (*NestedKey*) - where to write the padding mask.
Defaults to `"action_is_pad"`.
- **time_dim** (*int*) - the time dimension of the action tensor (the action
dimension must come right after it). Defaults to `-2`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs.transforms import ActionChunkTransform
>>> # for each step t the chunk gathers a[t], a[t+1], a[t+2], repeating
>>> # the last action past the end of the window (masked by action_is_pad)
>>> t = ActionChunkTransform(chunk_size=3)
>>> td = TensorDict(
... {"action": torch.arange(4).view(1, 4, 1).float()}, batch_size=[1, 4]
... )
>>> td = t(td)
>>> td["action_chunk"][0, :, :, 0]
tensor([[0., 1., 2.],
 [1., 2., 3.],
 [2., 3., 3.],
 [3., 3., 3.]])
>>> td["action_is_pad"][0]
tensor([[False, False, False],
 [False, False, False],
 [False, False, True],
 [False, True, True]])
>>> # on a replay buffer: extend with raw [T, action_dim] trajectory
>>> # windows (stored as-is), the chunks are built on the sample path
>>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(8),
... transform=ActionChunkTransform(chunk_size=3),
... batch_size=2,
... )
>>> windows = TensorDict(
... {"action": torch.randn(8, 4, 1)}, batch_size=[8]
... ) # 8 trajectory windows of T=4 steps each
>>> indices = rb.extend(windows)
>>> rb.sample()["action_chunk"].shape # [batch, T, chunk_size, action_dim]
torch.Size([2, 4, 3, 1])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionChunkTransform.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```