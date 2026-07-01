# QValueActor

*class*torchrl.modules.QValueActor(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#QValueActor)

A Q-Value actor class.

This class appends a [`QValueModule`](torchrl.modules.QValueModule.html#torchrl.modules.QValueModule) after the input module
such that the action values are used to select an action.

Parameters:

**module** (*nn.Module*) - a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) used to map the input to
the output parameter space. If the class provided is not compatible
with [`tensordict.nn.TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase), it will be
wrapped in a [`tensordict.nn.TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) with
`in_keys` indicated by the following keyword argument.

Keyword Arguments:

- **in_keys** (*iterable**of**str**,**optional*) - If the class provided is not
compatible with [`tensordict.nn.TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase), this
list of keys indicates what observations need to be passed to the
wrapped module to get the action values.
Defaults to `["observation"]`.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - Keyword-only argument.
Specs of the output tensor. If the module
outputs multiple output tensors,
spec characterize the space of the first output tensor.
- **safe** (*bool*) - Keyword-only argument.
If `True`, the value of the output is checked against the
input spec. Out-of-domain sampling can
occur because of exploration policies or numerical under/overflow
issues. If this value is out of bounds, it is projected back onto the
desired space using the `TensorSpec.project`
method. Default is `False`.
- **action_space** (*str**,**optional*) - Action space. Must be one of
`"one-hot"`, `"mult-one-hot"`, `"binary"` or `"categorical"`.
This argument is exclusive with `spec`, since `spec`
conditions the action_space.
- **action_value_key** (*str**or**tuple**of**str**,**optional*) - if the input module
is a [`tensordict.nn.TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) instance, it must
match one of its output keys. Otherwise, this string represents
the name of the action-value entry in the output tensordict.
- **action_key** (*str**or**tuple**of**str**,**optional*) - The output key for the selected
action. Defaults to `"action"`.
- **chosen_action_value_key** (*str**or**tuple**of**str**,**optional*) - The output key for
the selected action value. Defaults to `"chosen_action_value"`.
- **action_mask_key** (*str**or**tuple**of**str**,**optional*) - The input key
representing the action mask. Defaults to `"None"` (equivalent to no masking).

Note

`out_keys` cannot be passed. If the module is a [`tensordict.nn.TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)
instance, the out_keys will be updated accordingly. For regular
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) instance, the triplet `["action", action_value_key, "chosen_action_value"]`
will be used.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch import nn
>>> from torchrl.data import OneHot
>>> from torchrl.modules.tensordict_module.actors import QValueActor
>>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
>>> # with a regular nn.Module
>>> module = nn.Linear(4, 4)
>>> action_spec = OneHot(4)
>>> qvalue_actor = QValueActor(module=module, spec=action_spec)
>>> td = qvalue_actor(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
 action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=None,
 is_shared=False)
>>> # with a TensorDictModule
>>> td = TensorDict({'obs': torch.randn(5, 4)}, [5])
>>> module = TensorDictModule(lambda x: x, in_keys=["obs"], out_keys=["action_value"])
>>> action_spec = OneHot(4)
>>> qvalue_actor = QValueActor(module=module, spec=action_spec)
>>> td = qvalue_actor(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
 action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=None,
 is_shared=False)
```