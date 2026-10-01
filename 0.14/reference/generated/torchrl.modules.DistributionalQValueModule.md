# DistributionalQValueModule

*class*torchrl.modules.DistributionalQValueModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DistributionalQValueModule)

Distributional Q-Value hook for Q-value policies.

This module processes a tensor containing action value logits into is argmax
component (i.e. the resulting greedy action), following a given
action space (one-hot, binary or categorical).
It works with both tensordict and regular tensors.

The input action value is expected to be the result of a log-softmax
operation.

For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
[https://arxiv.org/pdf/1707.06887.pdf](https://arxiv.org/pdf/1707.06887.pdf)

Parameters:

- **action_space** (*str**,**optional*) - Action space. Must be one of
`"one-hot"`, `"mult-one-hot"`, `"binary"` or `"categorical"`.
This argument is exclusive with `spec`, since `spec`
conditions the action_space.
- **support** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - support of the action values.
- **action_value_key** (*str**or**tuple**of**str**,**optional*) - The input key
representing the action value. Defaults to `"action_value"`.
- **action_mask_key** (*str**or**tuple**of**str**,**optional*) - The input key
representing the action mask. Defaults to `"None"` (equivalent to no masking).
- **out_keys** (*list**of**str**or**tuple**of**str**,**optional*) - The output keys
representing the actions and action values.
Defaults to `["action", "action_value"]`.
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

Examples

```
>>> from tensordict import TensorDict
>>> torch.manual_seed(0)
>>> action_space = "categorical"
>>> action_value_key = "my_action_value"
>>> support = torch.tensor([-1, 0.0, 1.0]) # the action value is between -1 and 1
>>> actor = DistributionalQValueModule(action_space, support=support, action_value_key=action_value_key)
>>> # This module works with both tensordict and regular tensors:
>>> value = torch.full((3, 4), -100)
>>> # the first bin (-1) of the first action is high: there's a high chance that it has a low value
>>> value[0, 0] = 0
>>> # the second bin (0) of the second action is high: there's a high chance that it has an intermediate value
>>> value[1, 1] = 0
>>> # the third bin (0) of the this action is high: there's a high chance that it has an high value
>>> value[2, 2] = 0
>>> actor(my_action_value=value)
(tensor(2), tensor([[ 0, -100, -100, -100],
 [-100, 0, -100, -100],
 [-100, -100, 0, -100]]))
>>> actor(value)
(tensor(2), tensor([[ 0, -100, -100, -100],
 [-100, 0, -100, -100],
 [-100, -100, 0, -100]]))
>>> actor(TensorDict({action_value_key: value}, []))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 my_action_value: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

forward(*tensordict: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DistributionalQValueModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.