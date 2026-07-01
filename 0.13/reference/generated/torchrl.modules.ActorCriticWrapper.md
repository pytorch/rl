# ActorCriticWrapper

*class*torchrl.modules.ActorCriticWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticWrapper)

Actor-value operator without common module.

This class wraps together an actor and a value model that do not share a common observation embedding network:

[![../../_images/aafig-5b1c51d6da7f2229a6c42592c838f793bf136146.svg](../../_images/aafig-5b1c51d6da7f2229a6c42592c838f793bf136146.svg)](../../_images/aafig-5b1c51d6da7f2229a6c42592c838f793bf136146.svg)

To facilitate the workflow, this class comes with a get_policy_operator() and get_value_operator() methods, which
will both return a standalone TDModule with the dedicated functionality.

Parameters:

- **policy_operator** (*TensorDictModule*) - a policy operator that reads the hidden variable and returns an action
- **value_operator** (*TensorDictModule*) - a value operator, that reads the hidden variable and returns a value

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import (
... ActorCriticWrapper,
... ProbabilisticActor,
... NormalParamExtractor,
... TanhNormal,
... ValueOperator,
... )
>>> action_module = TensorDictModule(
... nn.Sequential(torch.nn.Linear(4, 8), NormalParamExtractor()),
... in_keys=["observation"],
... out_keys=["loc", "scale"],
... )
>>> td_module_action = ProbabilisticActor(
... module=action_module,
... in_keys=["loc", "scale"],
... distribution_class=TanhNormal,
... return_log_prob=True,
... )
>>> module_value = torch.nn.Linear(4, 1)
>>> td_module_value = ValueOperator(
... module=module_value,
... in_keys=["observation"],
... )
>>> td_module = ActorCriticWrapper(td_module_action, td_module_value)
>>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
>>> td_clone = td_module(td.clone())
>>> print(td_clone)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 state_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> td_clone = td_module.get_policy_operator()(td.clone())
>>> print(td_clone) # no value
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> td_clone = td_module.get_value_operator()(td.clone())
>>> print(td_clone) # no action
TensorDict(
 fields={
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 state_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

get_policy_head() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)

Returns a standalone policy operator that maps an observation to an action.

get_policy_operator() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticWrapper.get_policy_operator)

Returns a standalone policy operator that maps an observation to an action.

get_value_head() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)

Returns a standalone value network operator that maps an observation to a value estimate.

get_value_operator() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticWrapper.get_value_operator)

Returns a standalone value network operator that maps an observation to a value estimate.