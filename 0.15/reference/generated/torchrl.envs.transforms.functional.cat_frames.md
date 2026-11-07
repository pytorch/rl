# cat_frames

*class*torchrl.envs.transforms.functional.cat_frames(*tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *N: int*, *dim: int*, ***, *padding: Literal['same', 'constant'] = 'same'*, *padding_value: float = 0.0*, *time_dim: int = -1*, *done_mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*)[[source]](../../_modules/torchrl/envs/transforms/functional.html#cat_frames)

Stacks a sliding window of `N` successive frames along `dim`.

This is the pure, stateless core of the
[`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames) transform (the PyTorch
`F.x` / `nn.X` split): [`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames)
delegates its offline / replay-buffer (contiguous trajectory slice)
windowing to this function so that the two stay byte-for-byte identical.

For every position `t` along `time_dim`, the `N` frames
`[t - N + 1, ..., t]` are concatenated along `dim`. The first `N - 1`
positions of a trajectory have fewer than `N` real frames; the missing
frames are filled according to `padding`. This matches the offline
behavior of [`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames); see the
"Examples" of that class for the online (stateful, per-step) usage.

It was first proposed in "Playing Atari with Deep Reinforcement Learning"
([https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)).

Parameters:

- **tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the frames to stack. One of its dimensions
(`time_dim`) is the time axis along which the sliding window
moves; `dim` is the (channel/feature) axis along which the
`N` frames are concatenated.
- **N** (*int*) - number of successive frames to concatenate.
- **dim** (*int*) - the dimension along which the frames are concatenated.
Must be negative so that it is invariant to leading batch
dimensions. The size of `tensor` along `dim` is multiplied by
`N` in the output.

Keyword Arguments:

- **padding** (*str**,**optional*) - the padding method, one of `"same"` or
`"constant"`. With `"same"` (default) the first real frame of
the trajectory is repeated; with `"constant"` the missing frames
are filled with `padding_value`.
- **padding_value** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - the value used to pad when
`padding="constant"`. Defaults to `0`.
- **time_dim** (*int**,**optional*) - the dimension of `tensor` that holds the
time axis. Must be negative. Defaults to `-1`.
- **done_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - an optional boolean mask flagging,
for each sliding window, which of its `N` positions reach across
a trajectory boundary (and must therefore be padded). Its shape is
`(*batch, time, N)` where `time` matches the size of `tensor`
along `time_dim`. When `None` (default), the input is treated as
a single trajectory and only the leading `N - 1` start-of-sequence
frames are padded. [`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames)
builds this mask from the environment `done` signal.

Returns:

a tensor identical to `tensor` except that its size
along `dim` is multiplied by `N` (the concatenated window) and its
dtype / device are preserved.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Examples

```
>>> import torch
>>> from torchrl.envs.transforms.functional import cat_frames
>>> # a single trajectory of 4 frames, each a length-2 feature vector,
>>> # stacked over a window of N=3 along the feature dim (-1).
>>> frames = torch.arange(8.0).view(4, 2)
>>> frames
tensor([[0., 1.],
 [2., 3.],
 [4., 5.],
 [6., 7.]])
>>> out = cat_frames(frames, N=3, dim=-1, time_dim=-2, padding="constant")
>>> out.shape
torch.Size([4, 6])
>>> out
tensor([[0., 0., 0., 0., 0., 1.],
 [0., 0., 0., 1., 2., 3.],
 [0., 1., 2., 3., 4., 5.],
 [2., 3., 4., 5., 6., 7.]])
```

Note

This functional covers the **offline** (contiguous trajectory
slice) windowing used by
[`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames). The transform's
**online** path (per-[`step()`](torchrl.envs.EnvBase.html#id4) buffer
accumulation) is inherently stateful and is not expressed as a pure
function.

See also

[`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames).