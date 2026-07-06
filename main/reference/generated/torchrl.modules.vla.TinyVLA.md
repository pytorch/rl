# TinyVLA

*class*torchrl.modules.vla.TinyVLA(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/vla/models.html#TinyVLA)

A tiny, dependency-free reference VLA policy for CI and tutorials.

`TinyVLA` fuses a small convolutional image encoder, an optional
proprioceptive-state MLP, and a hashed language-instruction embedding into a
trunk that feeds either a continuous action-chunk head or a discrete
action-token head (see [`VLAWrapperBase`](torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase)). It is
intentionally small and CPU-friendly - a stand-in to exercise the VLA data
pipeline, losses and collectors end-to-end, **not** a competitive policy.

The language instruction is embedded by hashing the instruction string to an
embedding-table index (a deterministic, tokenizer-free stand-in), so the
policy is genuinely language-conditioned without any external dependency.

Note

`TinyVLA` expects observations with a single leading batch dimension
(`image` shaped `[B, C, H, W]`). When training on chunked windows,
flatten the time dimension into the batch first.

Keyword Arguments:

- **action_dim** (*int*) - the dimensionality of a single action.
- **chunk_size** (*int*) - the action-chunk horizon `H`.
- **action_head** (*str*) - `"continuous"` (default) or `"tokens"`.
- **vocab_size** (*int*) - action-token bins per dimension (token head).
Defaults to `256`.
- **use_state** (*bool*) - whether to read the proprioceptive state.
Defaults to `True`.
- **hidden_dim** (*int*) - width of the fused trunk. Defaults to `128`.
- **text_vocab** (*int*) - size of the hashed instruction embedding table.
Defaults to `256`.
- **text_dim** (*int*) - instruction-embedding dimension. Defaults to `32`.
- **default_interaction_type** (*InteractionType*) - token-head readout when no
exploration context is active (`RANDOM` samples, else argmax);
the forward otherwise follows the ambient
`exploration_type()`. Defaults to
`InteractionType.DETERMINISTIC`. See
[`VLAWrapperBase`](torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase).
- **mode** (*str**,**optional*) - backward-compatible alias for
`default_interaction_type`. Defaults to `None`.
- **device** (*DEVICE_TYPING**,**optional*) - device to move the parameters to.
- **return_vla_action_container** (*bool*) - whether to write the structured
VLAAction container in addition to its TensorDict fields. Defaults
to `True`.

Examples

```
>>> import torch
>>> from tensordict import NonTensorStack, TensorDict
>>> from torchrl.modules.vla import TinyVLA
>>> policy = TinyVLA(action_dim=7, chunk_size=4)
>>> td = TensorDict(
... {
... "observation": {
... "image": torch.zeros(2, 3, 16, 16, dtype=torch.uint8),
... "state": torch.zeros(2, 5),
... },
... "language_instruction": NonTensorStack("pick", "place"),
... },
... batch_size=[2],
... )
>>> policy(td)["vla_action", "chunk"].shape
torch.Size([2, 4, 7])
```