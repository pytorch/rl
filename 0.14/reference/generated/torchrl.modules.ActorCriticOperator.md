# ActorCriticOperator

*class*torchrl.modules.ActorCriticOperator(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticOperator)

Actor-critic operator.

This class wraps together an actor and a value model that share a common
observation embedding network:

[![../../_images/aafig-79381044a8773741ed8c83c5de90ab4def5c10b2.svg](../../_images/aafig-79381044a8773741ed8c83c5de90ab4def5c10b2.svg)](../../_images/aafig-79381044a8773741ed8c83c5de90ab4def5c10b2.svg)

Note

For a similar class that returns an action and a state-value \(V(s)\)
see [`ActorValueOperator`](torchrl.modules.ActorValueOperator.html#torchrl.modules.ActorValueOperator).

To facilitate the workflow, this class comes with a get_policy_operator() method, which
will both return a standalone TDModule with the dedicated functionality. The get_critic_operator will return the
parent object, as the value is computed based on the policy output.

Parameters:

- **common_operator** (*TensorDictModule*) - a common operator that reads
observations and produces a hidden variable
- **policy_operator** (*TensorDictModule*) - a policy operator that reads the
hidden variable and returns an action
- **value_operator** (*TensorDictModule*) - a value operator, that reads the
hidden variable and returns a value

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import ProbabilisticActor
>>> from torchrl.modules import ValueOperator, TanhNormal, ActorCriticOperator, NormalParamExtractor, MLP
>>> module_hidden = torch.nn.Linear(4, 4)
>>> td_module_hidden = SafeModule(
... module=module_hidden,
... in_keys=["observation"],
... out_keys=["hidden"],
... )
>>> module_action = nn.Sequential(torch.nn.Linear(4, 8), NormalParamExtractor())
>>> module_action = TensorDictModule(module_action, in_keys=["hidden"], out_keys=["loc", "scale"])
>>> td_module_action = ProbabilisticActor(
... module=module_action,
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=TanhNormal,
... return_log_prob=True,
... )
>>> module_value = MLP(in_features=8, out_features=1, num_cells=[])
>>> td_module_value = ValueOperator(
... module=module_value,
... in_keys=["hidden", "action"],
... out_keys=["state_action_value"],
... )
>>> td_module = ActorCriticOperator(td_module_hidden, td_module_action, td_module_value)
>>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
>>> td_clone = td_module(td.clone())
>>> print(td_clone)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 state_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> td_clone = td_module.get_policy_operator()(td.clone())
>>> print(td_clone) # no value
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> td_clone = td_module.get_critic_operator()(td.clone())
>>> print(td_clone) # no action
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 state_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

get_critic_operator() → [TensorDictModuleWrapper](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleWrapper.html#tensordict.nn.TensorDictModuleWrapper)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticOperator.get_critic_operator)

Returns a standalone critic network operator that maps a state-action pair to a critic estimate.

get_policy_head() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticOperator.get_policy_head)

Returns the policy head.

get_value_head() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticOperator.get_value_head)

Returns the value head.

get_value_operator() → [TensorDictModuleWrapper](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleWrapper.html#tensordict.nn.TensorDictModuleWrapper)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ActorCriticOperator.get_value_operator)

Returns a standalone value network operator that maps an observation to a value estimate.