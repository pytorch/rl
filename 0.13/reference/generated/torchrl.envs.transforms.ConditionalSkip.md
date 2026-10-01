# ConditionalSkip

*class*torchrl.envs.transforms.ConditionalSkip(*cond: Callable[[[TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)], bool | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#ConditionalSkip)

A transform that skips steps in the env if certain conditions are met.

This transform writes the result of cond(tensordict) in the "_step" entry of the
tensordict passed as input to the TransformedEnv.base_env._step method.
If the base_env is not batch-locked (generally speaking, it is stateless), the tensordict is
reduced to its element that need to go through the step.
If it is batch-locked (generally speaking, it is stateful), the step is skipped altogether if no
value in "_step" is `True`. Otherwise, it is trusted that the environment will account for the
"_step" signal accordingly.

Note

The skip will affect transforms that modify the environment output too, i.e., any transform
that is to be executed on the tensordict returned by [`step()`](torchrl.envs.EnvBase.html#id4) will be
skipped if the condition is met. To palliate this effect if it is not desirable, one can wrap
the transformed env in another transformed env, since the skip only affects the first-degree parent
of the `ConditionalSkip` transform. See example below.

Parameters:

**cond** (*Callable**[**[**TensorDictBase**]**,**bool**|*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*]*) - a callable for the tensordict input
that checks whether the next env step must be skipped (True = skipped, False = execute
env.step).

Examples

```
>>> import torch
>>>
>>> from torchrl.envs import GymEnv
>>> from torchrl.envs.transforms.transforms import ConditionalSkip, StepCounter, TransformedEnv, Compose
>>>
>>> torch.manual_seed(0)
>>>
>>> base_env = TransformedEnv(
... GymEnv("Pendulum-v1"),
... StepCounter(step_count_key="inner_count"),
... )
>>> middle_env = TransformedEnv(
... base_env,
... Compose(
... StepCounter(step_count_key="middle_count"),
... ConditionalSkip(cond=lambda td: td["step_count"] % 2 == 1),
... ),
... auto_unwrap=False) # makes sure that transformed envs are properly wrapped
>>> env = TransformedEnv(
... middle_env,
... StepCounter(step_count_key="step_count"),
... auto_unwrap=False)
>>> env.set_seed(0)
>>>
>>> r = env.rollout(10)
>>> print(r["observation"])
tensor([[-0.9670, -0.2546, -0.9669],
 [-0.9802, -0.1981, -1.1601],
 [-0.9802, -0.1981, -1.1601],
 [-0.9926, -0.1214, -1.5556],
 [-0.9926, -0.1214, -1.5556],
 [-0.9994, -0.0335, -1.7622],
 [-0.9994, -0.0335, -1.7622],
 [-0.9984, 0.0561, -1.7933],
 [-0.9984, 0.0561, -1.7933],
 [-0.9895, 0.1445, -1.7779]])
>>> print(r["inner_count"])
tensor([[0],
 [1],
 [1],
 [2],
 [2],
 [3],
 [3],
 [4],
 [4],
 [5]])
>>> print(r["middle_count"])
tensor([[0],
 [1],
 [1],
 [2],
 [2],
 [3],
 [3],
 [4],
 [4],
 [5]])
>>> print(r["step_count"])
tensor([[0],
 [1],
 [2],
 [3],
 [4],
 [5],
 [6],
 [7],
 [8],
 [9]])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#ConditionalSkip.forward)

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