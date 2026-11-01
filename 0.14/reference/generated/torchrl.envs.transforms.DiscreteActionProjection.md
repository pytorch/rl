# DiscreteActionProjection

*class*torchrl.envs.transforms.DiscreteActionProjection(*num_actions_effective: int*, *max_actions: int*, *action_key: NestedKey = 'action'*, *include_forward: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#DiscreteActionProjection)

Projects discrete actions from a high dimensional space to a low dimensional space.

Given a discrete action (from 1 to N) encoded as a one-hot vector and a
maximum action index num_actions (with num_actions < N), transforms the action such that
action_out is at most num_actions.

If the input action is > num_actions, it is being replaced by a random value
between 0 and num_actions-1. Otherwise the same action is kept.
This is intended to be used with policies applied over multiple discrete
control environments with different action space.

A call to DiscreteActionProjection.forward (eg from a replay buffer or in a
sequence of nn.Modules) will call the transform num_actions_effective -> max_actions
on the `"in_keys"`, whereas a call to _call will be ignored. Indeed,
transformed envs are instructed to update the input keys only for the inner
base_env, but the original input keys will remain unchanged.

Parameters:

- **num_actions_effective** (*int*) - max number of action considered.
- **max_actions** (*int*) - maximum number of actions that this module can read.
- **action_key** (*NestedKey**,**optional*) - key name of the action. Defaults to "action".
- **include_forward** (*bool**,**optional*) - if `True`, a call to forward will also
map the action from one domain to the other when the module is called
by a replay buffer or an nn.Module chain. Defaults to True.

Examples

```
>>> torch.manual_seed(0)
>>> N = 3
>>> M = 2
>>> action = torch.zeros(N, dtype=torch.long)
>>> action[-1] = 1
>>> td = TensorDict({"action": action}, [])
>>> transform = DiscreteActionProjection(num_actions_effective=M, max_actions=N)
>>> _ = transform.inv(td)
>>> print(td.get("action"))
tensor([1])
```

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#DiscreteActionProjection.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform