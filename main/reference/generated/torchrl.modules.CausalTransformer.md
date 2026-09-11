# CausalTransformer

*class*torchrl.modules.CausalTransformer(*input_size: int*, *hidden_size: int*, *num_layers: int = 1*, ***, *num_heads: int*, *max_seq_len: int*, *dim_feedforward: int | None = None*, *dropout: float = 0.0*, *device=None*)[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#CausalTransformer)

A causal transformer backbone with matching windowed and cached-step semantics.

This is the reference implementation of the temporal-transformer backbone
contract consumed by [`TransformerModule`](torchrl.modules.TransformerModule.html#torchrl.modules.TransformerModule):

- `forward(features, positions, mask=None, kv_cache=None) -> (out, kv_cache)`
- `new_kv_cache(batch_size, device=None) -> kv_cache`
- `reset_kv_cache(kv_cache, mask) -> kv_cache`

together with `num_layers`, `num_heads`, `head_dim` and
`max_seq_len` attributes. The cache object is opaque to the module: the
backbone decides its layout, dtype and device and how a reset clears the
rows selected by a boolean mask over the batch. Any module honoring that
contract can be used in its place, including adapters over an inference
engine that keeps the cache in its own representation.

Two execution paths share the same parameters and produce the same
outputs: a window path processing `[B, T]` at once under a causal mask
(training), and a cached-step path attending against a fixed-shape
key/value cache (collection). Positions are always explicit inputs, which
is what keeps the two paths consistent across episode resets.

The reference cache is a `(k, v)` pair of shape `[B, num_layers,
num_heads, max_seq_len, head_dim]` allocated in the dtype of the
projection weights, so a module converted to `bfloat16` or `float64`
gets a matching cache. Under autocast the projected keys and values are
cast to the cache dtype on write and the cache to the query dtype on
read. Cached entries are detached: the cached-step path is inference
only.

Parameters:

- **input_size** (*int*) - number of input features.
- **hidden_size** (*int*) - dimension of the residual stream. Must be divisible
by `num_heads`.
- **num_layers** (*int**,**optional*) - number of transformer blocks. Defaults to
`1`.

Keyword Arguments:

- **num_heads** (*int*) - number of attention heads.
- **max_seq_len** (*int*) - maximum episode length; sets the positional
embedding table and the cache size. Episodes longer than this
raise an error (sliding-window semantics are deliberately not
implemented).
- **dim_feedforward** (*int**,**optional*) - hidden dimension of the per-block
MLP. Defaults to `4 * hidden_size`.
- **dropout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - dropout probability in the block MLPs.
Defaults to `0.0`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to build the parameters on.

Examples

```
>>> import torch
>>> net = CausalTransformer(3, 16, 2, num_heads=4, max_seq_len=10)
>>> features = torch.randn(2, 5, 3)
>>> positions = torch.arange(5).expand(2, 5)
>>> out, _ = net(features, positions)
>>> out.shape
torch.Size([2, 5, 16])
>>> cache = net.new_kv_cache(2)
>>> step, cache = net(features[:, :1], positions[:, :1], kv_cache=cache)
>>> torch.allclose(step, out[:, :1], atol=1e-6)
True
```

forward(*features: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *positions: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *kv_cache: tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | None = None*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | None][[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#CausalTransformer.forward)

Run the backbone over a window or a single cached step.

Parameters:

- **features** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - `[B, T, input_size]` inputs. `T` must
be `1` when `kv_cache` is provided.
- **positions** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - `[B, T]` integer positions of each
step within its episode.
- **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - `[B, T, T]` boolean mask
(`True` = attend) for the window path; defaults to a plain
causal mask. Ignored on the cached-step path, where validity
is derived from `positions`.
- **kv_cache** (*tuple**of*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a cache from
`new_kv_cache()`. Providing it selects the cached-step
path; the cache is updated in place at `positions`.

Returns:

A tuple `(out, kv_cache)` with `out` of shape
`[B, T, hidden_size]` and `kv_cache` the updated cache on the
cached-step path (`None` on the window path).

new_kv_cache(*batch_size: int*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#CausalTransformer.new_kv_cache)

Allocate an empty key/value cache for `batch_size` streams.

Parameters:

**batch_size** (*int*) - number of concurrent streams (environments).

Keyword Arguments:

- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - where to allocate the cache.
Defaults to the device of the projection weights.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype of the cache. Pass the
compute dtype under autocast so cached keys and values are
stored as the projections produce them, without a conversion
on every step. Defaults to the dtype of the projection
weights.

Returns:

A `(k, v)` tuple of zero tensors of shape `[batch_size,
num_layers, num_heads, max_seq_len, head_dim]`.

*static*reset_kv_cache(*kv_cache: tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*, *mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#CausalTransformer.reset_kv_cache)

Clear the cache rows of the streams selected by `mask`.

Parameters:

- **kv_cache** (*tuple**of*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a cache from `new_kv_cache()`.
- **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean tensor of shape `[batch_size]`;
`True` rows are zeroed in place.

Returns:

The same `(k, v)` tuple.