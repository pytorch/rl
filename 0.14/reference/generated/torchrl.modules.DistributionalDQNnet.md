# DistributionalDQNnet

*class*torchrl.modules.DistributionalDQNnet(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#DistributionalDQNnet)

Distributional Deep Q-Network softmax layer.

This layer should be used in between a regular model that predicts the
action values and a distribution which acts on logits values.

Parameters:

- **in_keys** (*list**of**str**or**tuples**of**str*) - input keys to the log-softmax
operation. Defaults to `["action_value"]`.
- **out_keys** (*list**of**str**or**tuples**of**str*) - output keys to the log-softmax
operation. Defaults to `["action_value"]`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> net = DistributionalDQNnet()
>>> td = TensorDict({"action_value": torch.randn(10, 5)}, batch_size=[10])
>>> net(td)
TensorDict(
 fields={
 action_value: Tensor(shape=torch.Size([10, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
```

forward(*tensordict=None*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#DistributionalDQNnet.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.