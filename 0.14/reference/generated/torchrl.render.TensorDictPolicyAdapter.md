# TensorDictPolicyAdapter

*class*torchrl.render.TensorDictPolicyAdapter(*policy: Any*, *obs_key: Any*, *action_key: Any*)[[source]](../../_modules/torchrl/render/policy.html#TensorDictPolicyAdapter)

Adapts plain tensor policies to a TensorDict policy callable.

Parameters:

- **policy** - Policy object or callable.
- **obs_key** - Observation key used for tensor-only policies.
- **action_key** - Action key written when tensor actions are returned.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.render.policy import TensorDictPolicyAdapter
>>> def policy(obs):
... if not torch.is_tensor(obs):
... raise TypeError("expected tensor input")
... return obs + 1
>>> adapter = TensorDictPolicyAdapter(policy, "obs", ("agent", "action"))
>>> td = TensorDict({"obs": torch.zeros(1)}, [])
>>> adapter(td).get(("agent", "action"))
tensor([1.])
```