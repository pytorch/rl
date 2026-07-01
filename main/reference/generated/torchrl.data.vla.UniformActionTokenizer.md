# UniformActionTokenizer

*class*torchrl.data.vla.UniformActionTokenizer(*num_bins: int*, ***, *low: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *high: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action_dim: int | None = None*)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#UniformActionTokenizer)

Per-dimension uniform-bin action tokenizer (RT-2 / OpenVLA style).

Each action dimension is discretized into `num_bins` equal-width bins over
`[low, high]`; `encode()` returns the bin index and `decode()`
returns the bin center. The round-trip is lossy with error bounded by half a
bin width, `(high - low) / (2 * num_bins)`.

Parameters:

**num_bins** (*int*) - number of bins per action dimension.

Keyword Arguments:

- **low** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - per-dimension lower bound. Actions are
clamped to `[low, high]` before binning.
- **high** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - per-dimension upper bound.
- **action_dim** (*int**,**optional*) - action dimensionality. Required only when
`low`/`high` are scalars and you want a per-dimension shape.

Examples

```
>>> import torch
>>> from torchrl.data.vla import UniformActionTokenizer
>>> tok = UniformActionTokenizer(256, low=-1.0, high=1.0)
>>> tokens = tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
>>> tokens
tensor([ 0, 128, 255])
>>> torch.allclose(tok.decode(tokens), torch.tensor([-0.998, 0.002, 0.998]), atol=1e-2)
True
>>> tok.vocab_size
256
```

See also

[`RobotDatasetMetadata`](torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata) carries the
`action_low`/`action_high` bounds used by `from_metadata()`.

*property*action_dim*: int | None*

The per-dimension action size, or `None` for scalar bounds.

decode(*tokens: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#UniformActionTokenizer.decode)

Map token ids back to continuous actions `[..., action_dim]`.

encode(*actions: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#UniformActionTokenizer.encode)

Map continuous actions `[..., action_dim]` to token ids (`long`).

*classmethod*from_metadata(*metadata: [RobotDatasetMetadata](torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata)*, *num_bins: int*) → UniformActionTokenizer[[source]](../../_modules/torchrl/data/vla/tokenizers.html#UniformActionTokenizer.from_metadata)

Build from the `action_low`/`action_high` of a [`RobotDatasetMetadata`](torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata).

*property*vocab_size*: int*

Number of distinct token ids the tokenizer can emit per position.