# positions_from_is_init

*class*torchrl.modules.positions_from_is_init(*is_init: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#positions_from_is_init)

Compute per-token positions within each episode segment of a window.

Positions restart at `0` on every `is_init` flag. The first step of
the window is always treated as position `0`, so callers must pass
episode-aligned windows: [`TransformerModule`](torchrl.modules.TransformerModule.html#torchrl.modules.TransformerModule) validates that every
row of a training window starts with `is_init=True`.

Parameters:

**is_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean tensor of shape `[*batch, T]`
marking the first step of each episode.

Returns:

A `torch.long` tensor of shape `[*batch, T]` holding the position
of each step within its episode segment.

Examples

```
>>> is_init = torch.tensor([[True, False, True, False]])
>>> positions_from_is_init(is_init)
tensor([[0, 1, 0, 1]])
```