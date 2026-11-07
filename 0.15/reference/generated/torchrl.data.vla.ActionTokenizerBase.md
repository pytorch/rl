# ActionTokenizerBase

*class*torchrl.data.vla.ActionTokenizerBase(**args: Any*, ***kwargs: Any*)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#ActionTokenizerBase)

Base class for action tokenizers.

An action tokenizer maps continuous actions to discrete token ids and back,
so that autoregressive (RT-2 / OpenVLA-style) VLA policies can emit actions
through a language-model head and be trained with token cross-entropy.

A tokenizer operates element-wise over the trailing action dimension, so it
works unchanged on per-step actions `[*B, action_dim]` and on action
chunks `[*B, T, chunk, action_dim]`.

Subclasses implement `encode()`, `decode()` and the
`vocab_size` property.

decode(*tokens: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#ActionTokenizerBase.decode)

Map token ids back to continuous actions `[..., action_dim]`.

encode(*actions: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/vla/tokenizers.html#ActionTokenizerBase.encode)

Map continuous actions `[..., action_dim]` to token ids (`long`).

*property*vocab_size*: int*

Number of distinct token ids the tokenizer can emit per position.