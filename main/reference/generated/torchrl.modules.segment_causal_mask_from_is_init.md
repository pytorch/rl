# segment_causal_mask_from_is_init

*class*torchrl.modules.segment_causal_mask_from_is_init(*is_init: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#segment_causal_mask_from_is_init)

Build a block-diagonal causal attention mask from `is_init` flags.

Entry `[..., i, j]` is `True` (attend) iff `j <= i` and steps `i`
and `j` belong to the same episode segment, so attention never crosses
an episode boundary within a training window.

Parameters:

**is_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean tensor of shape `[*batch, T]`
marking the first step of each episode.

Returns:

A boolean tensor of shape `[*batch, T, T]` where `True` means
"may attend".

Examples

```
>>> is_init = torch.tensor([[False, True]])
>>> segment_causal_mask_from_is_init(is_init)
tensor([[[ True, False],
 [False, True]]])
```