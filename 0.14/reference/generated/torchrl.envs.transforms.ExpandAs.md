# ExpandAs

*class*torchrl.envs.transforms.ExpandAs(*in_key: NestedKey*, *ref_key: NestedKey*, *out_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#ExpandAs)

Expands one entry to the right to match a reference entry shape.

This is a transform wrapper around `tensordict.utils.expand_as_right()`.

Parameters:

- **in_key** (*NestedKey*) - key to expand.
- **ref_key** (*NestedKey*) - key used as shape reference.
- **out_key** (*NestedKey**,**optional*) - output key where the expanded tensor is
written. Defaults to `in_key`.

Examples

Expanding an environment-level `done` signal to the per-agent reward
shape in a VMAS environment:

```
>>> from torchrl.envs import TransformedEnv
>>> from torchrl.envs.libs.vmas import VmasEnv
>>> from torchrl.envs.transforms import ExpandAs
>>> base_env = VmasEnv(
... scenario="navigation",
... num_envs=16,
... continuous_actions=True,
... n_agents=3,
... )
>>> env = TransformedEnv(
... base_env,
... ExpandAs(
... in_key="done",
... ref_key=("agents", "reward"),
... ),
... )
>>> td = env.reset()
>>> td = env.rand_step(td)
>>> td["next", "done"].shape == td["next", "agents", "reward"].shape
True
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#ExpandAs.forward)

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

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#ExpandAs.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform