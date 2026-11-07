# ValueOperator

*class*torchrl.modules.ValueOperator(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ValueOperator)

General class for value functions in RL.

The ValueOperator class comes with default values for the in_keys and
out_keys arguments (["observation"] and ["state_value"] or
["state_action_value"], respectively and depending on whether the "action"
key is part of the in_keys list).

Parameters:

- **module** (*nn.Module*) - a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) used to map the input to
the output parameter space.
- **in_keys** (*iterable**of**str**,**optional*) - keys to be read from input
tensordict and passed to the module. If it
contains more than one element, the values will be passed in the
order given by the in_keys iterable.
Defaults to `["observation"]`.
- **out_keys** (*iterable**of**str*) - keys to be written to the input tensordict.
The length of out_keys must match the
number of tensors returned by the embedded module. Using "_" as a
key avoid writing tensor to output.
Defaults to `["state_value"]` or
`["state_action_value"]` if `"action"` is part of the `in_keys`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch import nn
>>> from torchrl.data import Unbounded
>>> from torchrl.modules import ValueOperator
>>> td = TensorDict({"observation": torch.randn(3, 4), "action": torch.randn(3, 2)}, [3,])
>>> class CustomModule(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = torch.nn.Linear(6, 1)
... def forward(self, obs, action):
... return self.linear(torch.cat([obs, action], -1))
>>> module = CustomModule()
>>> td_module = ValueOperator(
... in_keys=["observation", "action"], module=module
... )
>>> td = td_module(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 state_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```