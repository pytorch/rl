# ActionTokenizerTransform

*class*torchrl.envs.transforms.ActionTokenizerTransform(*tokenizer: [ActionTokenizerBase](torchrl.data.vla.ActionTokenizerBase.html#torchrl.data.vla.ActionTokenizerBase)*, ***, *in_key: NestedKey = 'action'*, *out_key: NestedKey = ('vla_action', 'tokens')*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionTokenizerTransform)

Encode and decode actions with an [`ActionTokenizerBase`](torchrl.data.vla.ActionTokenizerBase.html#torchrl.data.vla.ActionTokenizerBase).

A bidirectional action <-> token codec wrapping an action tokenizer (the
bins live in the tokenizer; no environment is needed to construct it).
Like any TorchRL transform it plugs onto a replay buffer or an environment
interchangeably:

- **forward** (`encode`): maps the continuous action (or action chunk) at
`in_key` to discrete token ids at `out_key` - e.g. building the token
training target for an autoregressive (RT-2 / OpenVLA-style) token VLA on
the replay-buffer sample path.
- **inverse** (`decode`): maps token ids at `out_key` back to a continuous
action at `in_key` - e.g. decoding the tokens a token-head policy emits,
on the environment action-input path, before the base env consumes them.
On a replay buffer the inverse is a no-op when the token entry is
absent, so extending with raw (untokenized) data is safe; attached to an
environment, missing tokens on the step path raise instead.

When attached to an environment, the policy-facing action spec is rewritten
to a [`Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical) over the tokenizer's vocabulary, so
the env advertises the token interface the policy is expected to produce
(the decoded continuous action is consumed by the base env internally).
Using the same tokenizer instance on the replay buffer (encode) and on the
env (decode) guarantees that training targets and execution share the exact
same binning.

Parameters:

**tokenizer** ([*ActionTokenizerBase*](torchrl.data.vla.ActionTokenizerBase.html#torchrl.data.vla.ActionTokenizerBase)) - the tokenizer to apply.

Keyword Arguments:

- **in_key** (*NestedKey*) - the continuous action. Defaults to `"action"`.
- **out_key** (*NestedKey*) - the discrete token ids. Defaults to
`("vla_action", "tokens")`. Pass `"action_tokens"` for the
flat compatibility key.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data.vla import UniformActionTokenizer
>>> from torchrl.envs.transforms import ActionTokenizerTransform
>>> tok = UniformActionTokenizer(256, low=-1.0, high=1.0)
>>> t = ActionTokenizerTransform(tok)
>>> td = t(TensorDict({"action": torch.tensor([[-1.0, 0.0, 1.0]])}, batch_size=[1]))
>>> td["vla_action", "tokens"]
tensor([[ 0, 128, 255]])
>>> # the inverse decodes tokens back to a continuous action
>>> back = t.inv(TensorDict({("vla_action", "tokens"): td["vla_action", "tokens"]}, batch_size=[1]))
>>> back["action"].shape
torch.Size([1, 3])
>>> # on a replay buffer: raw actions written through extend are stored
>>> # as-is and tokenized on the sample path
>>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(8),
... transform=ActionTokenizerTransform(tok),
... batch_size=2,
... )
>>> indices = rb.extend(
... TensorDict({"action": torch.rand(8, 3) * 2 - 1}, batch_size=[8])
... )
>>> rb.sample()["vla_action", "tokens"].shape
torch.Size([2, 3])
>>> # on an environment: the policy-facing action spec becomes the token
>>> # interface, and emitted tokens are decoded before the base env
>>> # consumes them
>>> from torchrl.envs import GymEnv, TransformedEnv
>>> tok_env = UniformActionTokenizer(256, low=-2.0, high=2.0) # Pendulum bounds
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), ActionTokenizerTransform(tok_env))
>>> env.full_action_spec["vla_action", "tokens"].shape
torch.Size([1])
>>> env.rollout(2)["vla_action", "tokens"].dtype
torch.int64
```

See also

[`ActionDiscretizer`](torchrl.envs.transforms.ActionDiscretizer.html#torchrl.envs.transforms.ActionDiscretizer) - the
env-only discretizer that derives its bins from the environment's
bounded `action_spec` (with configurable in-bin sampling strategies)
so a discrete-action policy can act on a continuous env. Use
`ActionDiscretizer` when the binning should follow the env spec; use
`ActionTokenizerTransform` when the binning is owned by a tokenizer
(dataset statistics, FAST/DCT-style codecs) that must be shared between
offline encoding and online decoding.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionTokenizerTransform.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionTokenizerTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform