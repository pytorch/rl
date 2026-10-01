# LastAction

*class*torchrl.envs.transforms.LastAction(*in_keys: Sequence[NestedKey] | NestedKey | None = None*, *out_keys: Sequence[NestedKey] | NestedKey | None = None*, ***, *default: Literal['zeros', 'nan'] | float | int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = 'zeros'*, *reset_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#LastAction)

Copies the last action into the next observation.

This is the action analogue of [`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames):
the policy can condition on the action taken at the previous step (delayed
control, recurrent policies, residual action heads). On each
[`step()`](torchrl.envs.EnvBase.html#id4) the action at time `t` is written
under `out_keys` in the `"next"` tensordict; on
[`reset()`](torchrl.envs.EnvBase.html#id1) the same keys are filled with a
default value (zeros, NaN, or a user-provided fill). On a batch-unlocked
parent the default is expanded by the runtime reset batch, preserving
the action feature shape.

`out_keys` are registered as [`Unbounded`](torchrl.data.Unbounded.html#torchrl.data.Unbounded)
observation specs with the action's shape, dtype and device, so reset
fills (zeros on a one-hot action, NaN on a bounded action) remain
in-spec.

Parameters:

- **in_keys** (*NestedKey**or**sequence**of**NestedKey**,**optional*) - keys pointing
to the actions to remember. Defaults to the parent environment's
[`action_keys`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.action_keys) when the transform is
attached, or `["action"]` otherwise.
- **out_keys** (*NestedKey**or**sequence**of**NestedKey**,**optional*) - destination
keys written into the observation. Defaults to each `in_keys`
entry with its last component replaced by `"last_action"`
(e.g. `"action"` -> `"last_action"`,
`("agents", "action")` -> `("agents", "last_action")`).

Keyword Arguments:

- **default** (*str**,**number**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - value used to fill
`out_keys` on [`reset()`](torchrl.envs.EnvBase.html#id1). `"zeros"`
(default) writes zeros matching the action spec; `"nan"` writes
NaNs (floating-point action specs only); a scalar is broadcast
with [`fill_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.fill_.html#torch.Tensor.fill_); a tensor is broadcast to the
action spec shape on the spec's device and dtype. Defaults to
`"zeros"`.
- **reset_key** (*NestedKey**,**optional*) - the reset key to be used as a
partial-reset indicator. Must be unique. If not provided, defaults
to the only reset key of the parent environment (if it has only
one) and raises an exception otherwise.

Examples

```
>>> from torchrl.envs import GymEnv, TransformedEnv
>>> from torchrl.envs.transforms import LastAction
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), LastAction())
>>> td = env.reset()
>>> td["last_action"]
tensor([0.])
>>> rollout = env.rollout(3)
>>> (rollout["next", "last_action"] == rollout["action"]).all()
tensor(True)
```

See also

[`CatFrames`](torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames) for stacking past
observations, [`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) for
marking episode starts, and
`LastActionConfig` for the
Hydra configuration.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_action.html#LastAction.forward)

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

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_action.html#LastAction.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform