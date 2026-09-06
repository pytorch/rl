# ActionChunkTransform

*class*torchrl.envs.transforms.ActionChunkTransform(*chunk_size: int*, ***, *action_key: NestedKey = 'action'*, *chunk_key: NestedKey = ('vla_action', 'chunk')*, *pad_key: NestedKey = 'action_is_pad'*, *time_dim: int = -2*, *done_key: NestedKey | None = 'done'*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionChunkTransform)

Build fixed-length action chunks from a trajectory window.

Action *chunking* is the defining trait of modern VLA policies (ACT,
OpenVLA-OFT, pi0, SmolVLA): instead of predicting a single action, the
policy predicts a short horizon `H` of future actions. This transform
turns a per-step action tensor `[*B, T, action_dim]` into the
corresponding training target `("vla_action", "chunk")` of shape
`[*B, T, H, action_dim]` - for each time step `t` it gathers the
actions `a[t], a[t+1], ..., a[t+H-1]` - together with a boolean
`action_is_pad` mask `[*B, T, H]` marking the steps that ran past the
end of the window (and were filled by repeating the last available action).

Internally this is a recipe over the generic transforms (the same pattern
as [`R3MTransform`](torchrl.envs.transforms.R3MTransform.html#torchrl.envs.transforms.R3MTransform)): an
[`UnsqueezeTransform`](torchrl.envs.transforms.UnsqueezeTransform.html#torchrl.envs.transforms.UnsqueezeTransform) opens the chunk dim
and a forward-looking [`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames)
(`future=True, padding="same", mask_key=...`) does the windowing, so
chunking shares one sliding-window implementation with frame stacking.

Changed in version 0.14: `ActionChunkTransform` is now a [`Compose`](torchrl.envs.transforms.Compose.html#torchrl.envs.transforms.Compose)
recipe over [`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames). The output is
unchanged, and additionally chunks become *boundary-aware* when the
sampled data carries its done state (see `done_key`).

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
`action_is_pad`. When the input additionally carries its done state at
`("next", done_key)`, chunks are also cut at the trajectory boundaries
*inside* a row: the steps past a done are padded (repeating the last
in-trajectory action) and flagged in `action_is_pad`, exactly like the
end of the window.

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
Defaults to `("vla_action", "chunk")`.
- **pad_key** (*NestedKey*) - where to write the padding mask.
Defaults to `"action_is_pad"`.
- **time_dim** (*int*) - the time dimension of the action tensor (the action
dimension must come right after it). Defaults to `-2`.
- **done_key** (*NestedKey**or**None*) -

the leaf done key: when the input
tensordict has a `("next", done_key)` entry (shaped like the
action without its trailing `action_dim`, with or without a
trailing singleton), chunks do not cross the trajectory boundaries
it marks. When the entry is absent, each row is treated as a
single contiguous trajectory (the pre-0.14 behavior). Pass
`None` to ignore the done state altogether.
Defaults to `"done"`.

New in version 0.14.

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
>>> td["vla_action", "chunk"][0, :, :, 0]
tensor([[0., 1., 2.],
 [1., 2., 3.],
 [2., 3., 3.],
 [3., 3., 3.]])
>>> td["action_is_pad"][0]
tensor([[False, False, False],
 [False, False, False],
 [False, False, True],
 [False, True, True]])
>>> # when the window carries its done state, chunks are also cut at
>>> # the trajectory boundary inside the window (here after step 1)
>>> td = TensorDict(
... {
... "action": torch.arange(4).view(1, 4, 1).float(),
... ("next", "done"): torch.tensor(
... [False, True, False, False]
... ).view(1, 4, 1),
... },
... batch_size=[1, 4],
... )
>>> t(td)["vla_action", "chunk"][0, :, :, 0]
tensor([[0., 1., 1.],
 [1., 1., 1.],
 [2., 3., 3.],
 [3., 3., 3.]])
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
>>> rb.sample()["vla_action", "chunk"].shape # [batch, T, chunk_size, action_dim]
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

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionChunkTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionChunkTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform