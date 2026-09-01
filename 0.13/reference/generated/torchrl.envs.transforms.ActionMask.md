# ActionMask

*class*torchrl.envs.transforms.ActionMask(*action_key: NestedKey = 'action'*, *mask_key: NestedKey = 'action_mask'*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionMask)

An adaptive action masker.

This transform is useful to ensure that randomly generated actions
respect legal actions, by masking the action specs.
It reads the mask from the input tensordict after the step is executed,
and adapts the mask of the finite action spec.

Note

This transform will fail when used without an environment.

Note

**MultiDiscrete action spaces with 2D masks (e.g., board games)**

When wrapping a Gym environment with a `MultiDiscrete` action space
(e.g., `MultiDiscrete([5, 5])`) and an `action_mask` observation whose
shape matches the `nvec` (e.g., shape `(5, 5)`), the [`GymWrapper`](torchrl.envs.GymWrapper.html#torchrl.envs.GymWrapper)
automatically converts the action space to a flattened `Categorical(n=25)`
or `OneHot(n=25)`. This allows the mask to represent all possible action
combinations (25 in this example) rather than independent sub-actions.

This is particularly useful for grid-based games where the mask indicates
which (row, column) positions are valid moves.

Parameters:

- **action_key** (*NestedKey**,**optional*) - the key where the action tensor can be found.
Defaults to `"action"`.
- **mask_key** (*NestedKey**,**optional*) - the key where the action mask can be found.
Defaults to `"action_mask"`.

Examples

```
>>> import torch
>>> from torchrl.data.tensor_specs import Categorical, Binary, Unbounded, Composite
>>> from torchrl.envs.transforms import ActionMask, TransformedEnv
>>> from torchrl.envs.common import EnvBase
>>> class MaskedEnv(EnvBase):
... def __init__(self, *args, **kwargs):
... super().__init__(*args, **kwargs)
... self.action_spec = Categorical(4)
... self.state_spec = Composite(action_mask=Binary(4, dtype=torch.bool))
... self.observation_spec = Composite(obs=Unbounded(3))
... self.reward_spec = Unbounded(1)
...
... def _reset(self, tensordict=None):
... td = self.observation_spec.rand()
... td.update(torch.ones_like(self.state_spec.rand()))
... return td
...
... def _step(self, data):
... td = self.observation_spec.rand()
... mask = data.get("action_mask")
... action = data.get("action")
... mask = mask.scatter(-1, action.unsqueeze(-1), 0)
...
... td.set("action_mask", mask)
... td.set("reward", self.reward_spec.rand())
... td.set("done", ~mask.any().view(1))
... return td
...
... def _set_seed(self, seed) -> None:
... pass
...
>>> torch.manual_seed(0)
>>> base_env = MaskedEnv()
>>> env = TransformedEnv(base_env, ActionMask())
>>> r = env.rollout(10)
>>> r["action_mask"]
tensor([[ True, True, True, True],
 [ True, True, False, True],
 [ True, True, False, False],
 [ True, False, False, False]])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionMask.forward)

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