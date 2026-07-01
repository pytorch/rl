# VLAWrapperBase

*class*torchrl.modules.vla.VLAWrapperBase(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase)

Base class for TensorDict-native Vision-Language-Action policies.

A VLA policy maps images, optional proprioceptive state, and a language
instruction to either a continuous action chunk or discrete action tokens.
Outputs are stored in a structured [`VLAAction`](torchrl.data.vla.VLAAction.html#torchrl.data.vla.VLAAction)
container under `"vla_action"` by default. Its fields are ordinary nested
TensorDict keys, e.g. `("vla_action", "chunk")` for continuous chunks.

Keyword Arguments:

- **action_dim** (*int*) - The dimensionality of a single action.
- **chunk_size** (*int*) - The action-chunk horizon.
- **action_head** (*str*) - `"continuous"` or `"tokens"`.
- **input_mode** (*str*) - `"canonical"` reads raw VLA keys. `"preprocessed"`
reads a [`VLAObservation`](torchrl.data.vla.VLAObservation.html#torchrl.data.vla.VLAObservation) or
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) from `observation_key`.
- **output_mode** (*str**,**optional*) - `"chunk"`, `"tokens"` or `"both"`.
Defaults to `"chunk"` for continuous heads and `"tokens"` for
token heads.
- **vocab_size** (*int**,**optional*) - Number of action-token bins, required for
token heads.
- **action_tokenizer** ([*ActionTokenizerBase*](torchrl.data.vla.ActionTokenizerBase.html#torchrl.data.vla.ActionTokenizerBase)*,**optional*) - Token/chunk codec used
when `output_mode` asks for both representations.
- **return_log_probs** (*bool**,**optional*) - Whether token `forward` writes
log-probabilities. Defaults to `True` for token heads.
- **return_logits** (*bool*) - Whether token `forward` writes `action_logits`.
- **logits_only** (*bool*) - Whether token `forward` returns logits without
sampling actions by default. A per-call `logits_only=True` argument
also enables this path.
- **log_probs_mode** (*str*) - `"sequence"` returns one summed log-probability
per sample; `"token"` returns per-token log-probabilities.
- **use_state** (*bool*) - Whether canonical mode reads the state key.
- **default_interaction_type** (*InteractionType*) - Token readout when no
exploration context is active.
- **mode** (*str**,**optional*) - Backward-compatible alias mapping `"sample"` to
`InteractionType.RANDOM` and `"greedy"` to deterministic.
- **inplace** (*bool**|**"empty"**|**None*) - Output TensorDict behavior. `True`
updates the input, `False` returns a new output TensorDict, and
`"empty"` returns an empty TensorDict populated with outputs.
- **num_samples** (*int**,**optional*) - Number of token samples to draw per input.

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
>>> out = policy(td)
>>> out["vla_action"].chunk.shape
torch.Size([2, 4, 7])
>>> out["vla_action", "chunk"].shape
torch.Size([2, 4, 7])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *tensordict_out: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *logits_only: bool = False*, ***kwargs*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

get_dist(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *tensordict_out: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *logits_key: NestedKey | None = None*, *mask_key: NestedKey | None = None*, ***kwargs*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.get_dist)

Return the action-token distribution for loss-time recomputation.

get_new_version(***kwargs*) → VLAWrapperBase[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.get_new_version)

Return a shallow wrapper copy with altered runtime parameters.

log_prob(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *action_key: NestedKey | None = None*, *log_probs_key: NestedKey | None = None*, ***kwargs*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.log_prob)

Recompute and write token log-probabilities for stored actions.

set_keys(***kwargs*) → VLAWrapperBase[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.set_keys)

Set the tensordict key names used by the policy.