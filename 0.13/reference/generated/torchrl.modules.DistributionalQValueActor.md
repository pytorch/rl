# DistributionalQValueActor

*class*torchrl.modules.DistributionalQValueActor(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DistributionalQValueActor)

A Distributional DQN actor class.

This class appends a [`QValueModule`](torchrl.modules.QValueModule.html#torchrl.modules.QValueModule) after the input module
such that the action values are used to select an action.

Parameters:

**module** (*nn.Module*) - a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) used to map the input to
the output parameter space.
If the module isn't of type [`torchrl.modules.DistributionalDQNnet`](torchrl.modules.DistributionalDQNnet.html#torchrl.modules.DistributionalDQNnet),
`DistributionalQValueActor` will ensure that a log-softmax
operation is applied to the action value tensor along dimension `-2`.
This can be deactivated by turning off the `make_log_softmax`
keyword argument.

Keyword Arguments:

- **in_keys** (*iterable**of**str**,**optional*) - keys to be read from input
tensordict and passed to the module. If it
contains more than one element, the values will be passed in the
order given by the in_keys iterable.
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
- **var_nums** (*int**,**optional*) - if `action_space = "mult-one-hot"`,
this value represents the cardinality of each
action component.
- **support** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - support of the action values.
- **action_space** (*str**,**optional*) - Action space. Must be one of
`"one-hot"`, `"mult-one-hot"`, `"binary"` or `"categorical"`.
This argument is exclusive with `spec`, since `spec`
conditions the action_space.
- **make_log_softmax** (*bool**,**optional*) - if `True` and if the module is not
of type [`torchrl.modules.DistributionalDQNnet`](torchrl.modules.DistributionalDQNnet.html#torchrl.modules.DistributionalDQNnet), a log-softmax
operation will be applied along dimension -2 of the action value tensor.
- **action_value_key** (*str**or**tuple**of**str**,**optional*) - if the input module
is a [`tensordict.nn.TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) instance, it must
match one of its output keys. Otherwise, this string represents
the name of the action-value entry in the output tensordict.
- **action_mask_key** (*str**or**tuple**of**str**,**optional*) - The input key
representing the action mask. Defaults to `"None"` (equivalent to no masking).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> from torch import nn
>>> from torchrl.data import OneHot
>>> from torchrl.modules import DistributionalQValueActor, MLP
>>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
>>> nbins = 3
>>> module = MLP(out_features=(nbins, 4), depth=2)
>>> # let us make sure that the output is a log-softmax
>>> module = TensorDictSequential(
... TensorDictModule(module, ["observation"], ["action_value"]),
... TensorDictModule(lambda x: x.log_softmax(-2), ["action_value"], ["action_value"]),
... )
>>> action_spec = OneHot(4)
>>> qvalue_actor = DistributionalQValueActor(
... module=module,
... spec=action_spec,
... support=torch.arange(nbins))
>>> td = qvalue_actor(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
 action_value: Tensor(shape=torch.Size([5, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=None,
 is_shared=False)
```