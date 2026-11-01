# Actor

*class*torchrl.modules.tensordict_module.Actor(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#Actor)

General class for deterministic actors in RL.

The Actor class comes with default values for the out_keys (`["action"]`)
and if the spec is provided but not as a
[`Composite`](torchrl.data.Composite.html#torchrl.data.Composite) object, it will be
automatically translated into `spec = Composite(action=spec)`.

Parameters:

- **module** (*nn.Module*) - a [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) used to map the input to
the output parameter space.
- **in_keys** (*iterable**of**str**,**optional*) - keys to be read from input
tensordict and passed to the module. If it
contains more than one element, the values will be passed in the
order given by the in_keys iterable.
Defaults to `["observation"]`.
- **out_keys** (*iterable**of**str*) - keys to be written to the input tensordict.
The length of out_keys must match the
number of tensors returned by the embedded module. Using `"_"` as a
key avoid writing tensor to output.
Defaults to `["action"]`.

Keyword Arguments:

- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - Keyword-only argument.
Specs of the output tensor. If the module
outputs multiple output tensors,
spec characterize the space of the first output tensor.
- **safe** (*bool*) - Keyword-only argument.
If `True`, the value of the output is checked against the
input spec. Out-of-domain sampling can
occur because of exploration policies or numerical under/overflow
issues. If this value is out of bounds, it is projected back onto the
desired space using the [`project()`](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.project)
method. Default is `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import Unbounded
>>> from torchrl.modules import Actor
>>> torch.manual_seed(0)
>>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
>>> action_spec = Unbounded(4)
>>> module = torch.nn.Linear(4, 4)
>>> td_module = Actor(
... module=module,
... spec=action_spec,
... )
>>> td_module(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> print(td.get("action"))
tensor([[-1.3635, -0.0340, 0.1476, -1.3911],
 [-0.1664, 0.5455, 0.2247, -0.4583],
 [-0.2916, 0.2160, 0.5337, -0.5193]], grad_fn=<AddmmBackward0>)
```

get_dist(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#Actor.get_dist)

Returns a Delta distribution centered at the deterministic action.

For deterministic actors, this returns a Delta distribution which has
log-probability 0 for the exact action and -inf for any other action.

Parameters:

**tensordict** (*TensorDictBase*) - input tensordict containing observations.

Returns:

A Delta distribution.

Return type:

torch.distributions.Distribution