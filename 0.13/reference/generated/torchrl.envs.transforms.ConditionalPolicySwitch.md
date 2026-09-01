# ConditionalPolicySwitch

*class*torchrl.envs.transforms.ConditionalPolicySwitch(*policy: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *condition: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], bool]*)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#ConditionalPolicySwitch)

A transform that conditionally switches between policies based on a specified condition.

This transform evaluates a condition on the data returned by the environment's step method.
If the condition is met, it applies a specified policy to the data. Otherwise, the data is
returned unaltered. This is useful for scenarios where different policies need to be applied
based on certain criteria, such as alternating turns in a game.

Parameters:

- **policy** (*Callable**[**[**TensorDictBase**]**,**TensorDictBase**]*) - The policy to be applied when the condition is met. This should be a callable that
takes a TensorDictBase and returns a TensorDictBase.
- **condition** (*Callable**[**[**TensorDictBase**]**,**bool**]*) - A callable that takes a TensorDictBase and returns a boolean or a tensor indicating
whether the policy should be applied.

Warning

This transform must have a parent environment.

Note

Ideally, it should be the last transform in the stack. If the policy requires transformed
data (e.g., images), and this transform is applied before those transformations, the policy will
not receive the transformed data.

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule as Mod
>>>
>>> from torchrl.envs import GymEnv, ConditionalPolicySwitch, Compose, StepCounter
>>> # Create a CartPole environment. We'll be looking at the obs: if the first element of the obs is greater than
>>> # 0 (left position) we do a right action (action=0) using the switch policy. Otherwise, we use our main
>>> # policy which does a left action.
>>> base_env = GymEnv("CartPole-v1", categorical_action_encoding=True)
>>>
>>> policy = Mod(lambda: torch.ones((), dtype=torch.int64), in_keys=[], out_keys=["action"])
>>> policy_switch = Mod(lambda: torch.zeros((), dtype=torch.int64), in_keys=[], out_keys=["action"])
>>>
>>> cond = lambda td: td.get("observation")[..., 0] >= 0
>>>
>>> env = base_env.append_transform(
... Compose(
... # We use two step counters to show that one counts the global steps, whereas the other
... # only counts the steps where the main policy is executed
... StepCounter(step_count_key="step_count_total"),
... ConditionalPolicySwitch(condition=cond, policy=policy_switch),
... StepCounter(step_count_key="step_count_main"),
... )
... )
>>>
>>> env.set_seed(0)
>>> torch.manual_seed(0)
>>>
>>> r = env.rollout(100, policy=policy)
>>> print("action", r["action"])
action tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> print("obs", r["observation"])
obs tensor([[ 0.0322, -0.1540, 0.0111, 0.3190],
 [ 0.0299, -0.1544, 0.0181, 0.3280],
 [ 0.0276, -0.1550, 0.0255, 0.3414],
 [ 0.0253, -0.1558, 0.0334, 0.3596],
 [ 0.0230, -0.1569, 0.0422, 0.3828],
 [ 0.0206, -0.1582, 0.0519, 0.4117],
 [ 0.0181, -0.1598, 0.0629, 0.4469],
 [ 0.0156, -0.1617, 0.0753, 0.4891],
 [ 0.0130, -0.1639, 0.0895, 0.5394],
 [ 0.0104, -0.1665, 0.1058, 0.5987],
 [ 0.0076, -0.1696, 0.1246, 0.6685],
 [ 0.0047, -0.1732, 0.1463, 0.7504],
 [ 0.0016, -0.1774, 0.1715, 0.8459],
 [-0.0020, 0.0150, 0.1884, 0.6117],
 [-0.0017, 0.2071, 0.2006, 0.3838]])
>>> print("obs'", r["next", "observation"])
obs' tensor([[ 0.0299, -0.1544, 0.0181, 0.3280],
 [ 0.0276, -0.1550, 0.0255, 0.3414],
 [ 0.0253, -0.1558, 0.0334, 0.3596],
 [ 0.0230, -0.1569, 0.0422, 0.3828],
 [ 0.0206, -0.1582, 0.0519, 0.4117],
 [ 0.0181, -0.1598, 0.0629, 0.4469],
 [ 0.0156, -0.1617, 0.0753, 0.4891],
 [ 0.0130, -0.1639, 0.0895, 0.5394],
 [ 0.0104, -0.1665, 0.1058, 0.5987],
 [ 0.0076, -0.1696, 0.1246, 0.6685],
 [ 0.0047, -0.1732, 0.1463, 0.7504],
 [ 0.0016, -0.1774, 0.1715, 0.8459],
 [-0.0020, 0.0150, 0.1884, 0.6117],
 [-0.0017, 0.2071, 0.2006, 0.3838],
 [ 0.0105, 0.2015, 0.2115, 0.5110]])
>>> print("total step count", r["step_count_total"].squeeze())
total step count tensor([ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27])
>>> print("total step with main policy", r["step_count_main"].squeeze())
total step with main policy tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → Any[[source]](../../_modules/torchrl/envs/transforms/_misc.html#ConditionalPolicySwitch.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```