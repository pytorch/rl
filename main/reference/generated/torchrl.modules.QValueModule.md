# QValueModule

*class*torchrl.modules.QValueModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#QValueModule)

Q-Value TensorDictModule for Q-value policies.

This module processes a tensor containing action value into is argmax
component (i.e. the resulting greedy action), following a given
action space (one-hot, binary or categorical).
It works with both tensordict and regular tensors.

Parameters:

- **action_space** (*str**,**optional*) - Action space. Must be one of
`"one-hot"`, `"mult-one-hot"`, `"binary"` or `"categorical"`.
This argument is exclusive with `spec`, since `spec`
conditions the action_space.
- **action_value_key** (*str**or**tuple**of**str**,**optional*) - The input key
representing the action value. Defaults to `"action_value"`.
- **action_mask_key** (*str**or**tuple**of**str**,**optional*) - The input key
representing the action mask. Defaults to `"None"` (equivalent to no masking).
- **out_keys** (*list**of**str**or**tuple**of**str**,**optional*) - The output keys
representing the actions, action values and chosen action value.
Defaults to `["action", "action_value", "chosen_action_value"]`.
- **var_nums** (*int**,**optional*) - if `action_space = "mult-one-hot"`,
this value represents the cardinality of each
action component.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - if provided, the specs of the action (and/or
other outputs). This is exclusive with `action_space`, as the spec
conditions the action space.
- **safe** (*bool*) - if `True`, the value of the output is checked against the
input spec. Out-of-domain sampling can
occur because of exploration policies or numerical under/overflow issues.
If this value is out of bounds, it is projected back onto the
desired space using the `TensorSpec.project`
method. Default is `False`.

Returns:

if the input is a single tensor, a triplet containing the chosen action,
the values and the value of the chose action is returned. If a tensordict
is provided, it is updated with these entries at the keys indicated by the
`out_keys` field.

Examples

```
>>> from tensordict import TensorDict
>>> action_space = "categorical"
>>> action_value_key = "my_action_value"
>>> actor = QValueModule(action_space, action_value_key=action_value_key)
>>> # This module works with both tensordict and regular tensors:
>>> value = torch.zeros(4)
>>> value[-1] = 1
>>> actor(my_action_value=value)
(tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
>>> actor(value)
(tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
>>> actor(TensorDict({action_value_key: value}, []))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 chosen_action_value: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 my_action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

forward(*tensordict: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#QValueModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.