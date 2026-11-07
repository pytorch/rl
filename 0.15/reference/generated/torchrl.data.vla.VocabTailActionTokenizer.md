# VocabTailActionTokenizer

*class*torchrl.data.vla.VocabTailActionTokenizer(*num_bins: int = 256*, ***, *full_vocab_size: int | None = None*, *norm_low: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *norm_high: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *norm_mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *gripper_binarize: bool = False*, *gripper_binarize_threshold: float = 0.0*, *gripper_invert: bool = False*)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#VocabTailActionTokenizer)

OpenVLA-style vocab-tail action tokenizer.

OpenVLA ([arXiv:2406.09246](https://arxiv.org/abs/2406.09246))
discretizes each normalized action dimension over the *edges* of
`num_bins` uniform bins spanning `[-1, 1]` and writes the result into
the last `num_bins` ids of the language-model vocabulary:
`full_token_id = vocab_size - digitize(action)`. Decoding maps a token
back to the corresponding bin center (there are `num_bins - 1` centers).
This tokenizer reproduces that exact mapping, with two id conventions:

- **window ids** (default, `full_vocab_size=None`): ids in
`[0, num_bins)` - the offset of the token inside the vocab-tail
window, `window_id = num_bins - digitize(action)`. This is the
convention of a token-head VLA policy emitting a `num_bins`-way
categorical per action dimension (e.g.
[`VLAWrapperBase`](torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase) with
`vocab_size=num_bins`).
- **full ids**: pass `full_vocab_size` (e.g. `32000` for LLaMA-2) to
use raw language-model token ids,
`full_id = full_vocab_size - digitize(action)`.

Optionally, dataset statistics (the `norm_stats` shipped with OpenVLA
checkpoints) un-normalize decoded actions to the environment's action
space - and normalize actions before encoding - via the affine q01/q99
map `a_env = 0.5 * (a + 1) * (q99 - q01 + 1e-8) + q01` applied to the
dimensions selected by `mask` (the gripper dimension is typically
excluded). To reproduce the reference MuJoCo action path exactly, decoding
with normalization statistics is performed in NumPy float64 on CPU; the
result is cast back to `float32` on the input tokens' device. See
`from_norm_stats()` and `decode()`.

Parameters:

**num_bins** (*int*) - number of bin edges per action dimension (the OpenVLA
convention; there are `num_bins - 1` bin centers). Defaults to
`256`.

Keyword Arguments:

- **full_vocab_size** (*int**,**optional*) - if provided, tokens are raw
language-model ids in `[full_vocab_size - num_bins,
full_vocab_size)` instead of window offsets. Defaults to
`None`.
- **norm_low** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-dimension lower statistics
(`q01`) for un-normalization. Defaults to `None` (no
normalization; actions live in `[-1, 1]`).
- **norm_high** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-dimension upper statistics
(`q99`).
- **norm_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - boolean mask of the dimensions to
(un-)normalize; unmasked dimensions pass through. Defaults to all
`True` when statistics are given.
- **gripper_binarize** (*bool**,**optional*) - if `True`, binarize unmasked
dimensions (usually gripper) to `-1` / `+1` after decoding.
Defaults to `False`.
- **gripper_binarize_threshold** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - threshold used for
gripper binarization: values strictly above this threshold map to
`+1`, the rest to `-1`. Defaults to `0.0`.
- **gripper_invert** (*bool**,**optional*) - if `True`, flip the sign of
unmasked dimensions after optional binarization. Defaults to
`False`.

Examples

```
>>> import torch
>>> from torchrl.data.vla import VocabTailActionTokenizer
>>> tok = VocabTailActionTokenizer(256)
>>> tokens = tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
>>> tokens
tensor([255, 128, 0])
>>> tok.decode(tokens)
tensor([-0.9961, 0.0000, 0.9961])
>>> # full LM-vocabulary ids (LLaMA-2)
>>> tok = VocabTailActionTokenizer(256, full_vocab_size=32000)
>>> tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
tensor([31999, 31872, 31744])
>>> tok.vocab_size
32000
```

See also

[`UniformActionTokenizer`](torchrl.data.vla.UniformActionTokenizer.html#torchrl.data.vla.UniformActionTokenizer) for the
plain bin-index codec used by toy token policies.

decode(*tokens: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#VocabTailActionTokenizer.decode)

Map token ids back to continuous actions `[..., action_dim]`.

When normalization statistics are set (see `from_norm_stats()`),
the de-tokenization and the q01/q99 un-normalization are computed in
NumPy float64 on CPU for bit-exact parity with the OpenVLA-OFT
reference implementation. This incurs a device-to-host round-trip
(and a host sync) on every call when `tokens` lives on an
accelerator; the result is moved back to `tokens.device` and cast
to the tokenizer's working dtype (`float32`).

Without statistics, decoding runs entirely on `tokens.device` and
returns bin centers in `[-1, 1]`.

encode(*actions: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#VocabTailActionTokenizer.encode)

Map continuous actions `[..., action_dim]` to token ids (`long`).

*classmethod*from_norm_stats(*norm_stats: dict*, *unnorm_key: str*, ***, *num_bins: int = 256*, *full_vocab_size: int | None = None*, *gripper_binarize: bool = False*, *gripper_binarize_threshold: float = 0.0*, *gripper_invert: bool = False*) → VocabTailActionTokenizer[[source]](../../_modules/torchrl/data/vla/tokenizers.html#VocabTailActionTokenizer.from_norm_stats)

Build from the `norm_stats` dictionary of an OpenVLA checkpoint.

Parameters:

- **norm_stats** (*dict*) - the checkpoint's normalization statistics
(`model.norm_stats`), mapping dataset keys to
`{"action": {"q01": ..., "q99": ..., "mask": ...}}`.
- **unnorm_key** (*str*) - the dataset key to use (e.g.
`"libero_spatial_no_noops"`).
- **num_bins** (*int**,**optional*) - number of bin edges. Defaults to `256`.
- **full_vocab_size** (*int**,**optional*) - raw language-model vocabulary size
when using full token ids. Defaults to `None`.
- **gripper_binarize** (*bool**,**optional*) - whether to binarize unmasked
gripper dimensions after decoding. Defaults to `False`.
- **gripper_binarize_threshold** (*float**,**optional*) - threshold used for
gripper binarization. Defaults to `0.0`.
- **gripper_invert** (*bool**,**optional*) - whether to invert unmasked gripper
dimensions after optional binarization. Defaults to `False`.

*property*vocab_size*: int*

Number of distinct token ids the tokenizer can emit per position.