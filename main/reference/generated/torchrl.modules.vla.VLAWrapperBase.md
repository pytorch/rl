# VLAWrapperBase

*class*torchrl.modules.vla.VLAWrapperBase(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase)

Base class for Vision-Language-Action policies.

A VLA policy maps multimodal robot observations - one or more camera
images, optional proprioceptive state, and a natural-language instruction -
to a short *action chunk*. This base owns the TensorDict key contract and
the `forward()` / `get_dist()` orchestration; concrete policies only
implement the prediction hooks `_predict_chunk()` (continuous head) and
`_predict_logits()` (discrete-token head).

Two action heads are supported via `action_head`:

- `"continuous"`: `forward()` writes a continuous action chunk of
shape `[*B, chunk_size, action_dim]` under `action_chunk`;
- `"tokens"`: `forward()` writes discrete action tokens
`[*B, chunk_size, action_dim]` under `action_tokens` and their
per-sample (sequence-level, summed over the chunk) log-probabilities
under `log_probs`; `get_dist()` returns
the token distribution for log-prob/entropy-based RL fine-tuning.

Keys are configurable through `set_keys()`. The wrapper is a
[`TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase), so it composes with the
standard collectors, losses and transforms.

Keyword Arguments:

- **action_dim** (*int*) - the dimensionality of a single action.
- **chunk_size** (*int*) - the action-chunk horizon `H`.
- **action_head** (*str*) - `"continuous"` (default) or `"tokens"`.
- **vocab_size** (*int**,**optional*) - number of action-token bins per dimension
(required for the `"tokens"` head).
- **use_state** (*bool*) - whether to read the proprioceptive state.
Defaults to `True`.
- **mode** (*str*) - `"greedy"` (default, argmax) or `"sample"` token
sampling for the `"tokens"` head (ignored by the continuous head).

Note

This base deliberately does **not** inherit from the text-generation
[`LLMWrapperBase`](torchrl.modules.llm.LLMWrapperBase.html#torchrl.modules.llm.LLMWrapperBase): a VLA policy emits robot
actions, not text, so it carries only the small multimodal-to-action
contract.

See also

[`TinyVLA`](torchrl.modules.vla.TinyVLA.html#torchrl.modules.vla.TinyVLA) (reference policy).

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

get_dist(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Independent](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.independent.Independent)[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.get_dist)

Return the action-token distribution.

Only defined for the `"tokens"` action head: a
`Categorical` over the vocabulary,
wrapped in `Independent` over the
`(chunk_size, action_dim)` token dims, so `log_prob` returns one
*sequence-level* log-probability per sample. This is the contract
PPO-style objectives expect: token RL fine-tuning works directly with
[`ClipPPOLoss`](torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) (pass
`critic_network=None`, `entropy_bonus=False` and remap the keys
via `set_keys`).

set_keys(***kwargs*) → VLAWrapperBase[[source]](../../_modules/torchrl/modules/vla/common.html#VLAWrapperBase.set_keys)

Set the tensordict key names used by the policy (see `_AcceptedKeys`).