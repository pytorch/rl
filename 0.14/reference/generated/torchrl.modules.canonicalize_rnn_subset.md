# canonicalize_rnn_subset

*class*torchrl.modules.canonicalize_rnn_subset(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *modules: Iterable[[LSTMModule](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) | [GRUModule](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule)]*, ***, *inplace: bool = False*)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#canonicalize_rnn_subset)

Canonicalize only the union of RNN keys used by `modules`.

Convenience wrapper around [`LSTMModule.canonicalize()`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule.canonicalize) /
[`GRUModule.canonicalize()`](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule.canonicalize) for pipelines that feed several recurrent
modules from the same TensorDict (e.g. a recurrent actor and a recurrent
critic). The union of every module's `canonical_keys` is collected,
canonicalized once, and merged back. Other leaves are untouched.

Parameters:

- **data** - TensorDict to canonicalize.
- **modules** - Iterable of [`LSTMModule`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) / [`GRUModule`](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule) whose
`canonical_keys` define the subset to canonicalize.
- **inplace** - When `True`, mutates `data` in place and returns it.
Defaults to `False`.

Returns:

A TensorDict with canonical layout on the RNN-relevant leaves.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import LSTMModule, canonicalize_rnn_subset
>>> actor = LSTMModule(input_size=3, hidden_size=4, in_key="obs",
... out_key="actor_h")
>>> critic = LSTMModule(input_size=3, hidden_size=4, in_key="obs",
... out_key="critic_h")
>>> td = TensorDict({"obs": torch.zeros(2, 5, 3)}, batch_size=[2, 5])
>>> canonicalize_rnn_subset(td, [actor, critic])["obs"].is_contiguous()
True
```