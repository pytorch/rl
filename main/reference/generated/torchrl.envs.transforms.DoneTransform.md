# DoneTransform

*class*torchrl.envs.transforms.DoneTransform(*in_keys: Sequence[NestedKey] | NestedKey | None = None*, *out_keys: Sequence[NestedKey] | NestedKey | None = None*, ***, *reward_key: NestedKey | None = None*, *done_keys: Sequence[NestedKey] | NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#DoneTransform)

Expands done flags to match the reward shape.

Multi-agent environments often expose a shared (environment-level) done
while rewards are per-agent. Value estimators such as GAE expect these
entries to share a trailing shape. This transform expands each done key
to the reward shape and writes the result under the reward group
(for example `("agents", "done")` when the reward key is
`("agents", "reward")`).
Remapped dones are observations and are not added to `env.done_keys`.

The transform can be appended to a `TransformedEnv`,
a collector (as `postproc`), or a replay buffer. When used as a collector
or replay-buffer transform, `forward()` expands entries under the
`"next"` sub-tensordict if that key is present.

Parameters:

- **in_keys** (*NestedKey**or**sequence**of**NestedKey**,**optional*) - done keys to
expand. Defaults to `("done", "terminated")`. A single NestedKey
is accepted. Mutually exclusive with `done_keys`.
- **out_keys** (*NestedKey**or**sequence**of**NestedKey**,**optional*) - destination
keys, one per `in_keys` entry. Defaults to the last component
of each input key placed under the reward group (e.g.
`("agents", "done")` if `reward_key` is `("agents", "reward")`
and the input key ends with `"done"`).

Keyword Arguments:

- **reward_key** (*NestedKey**,**optional*) - key of the reward used as the
expansion target. Defaults to `"reward"`. The default
`out_keys` are derived from this key's group unless
`out_keys` is provided.
- **done_keys** (*NestedKey**or**sequence**of**NestedKey**,**optional*) - alias of
`in_keys` kept for compatibility with the historical multi-agent
helper. Mutually exclusive with `in_keys`.

See also [`DoneTransformConfig`](torchrl.trainers.algorithms.configs.transforms.DoneTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.DoneTransformConfig).

Examples

Expand shared done flags onto the per-agent reward shape:

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs.transforms import DoneTransform
>>> n_envs, n_agents = 2, 3
>>> td = TensorDict(
... {
... "done": torch.tensor([[False], [True]]),
... "terminated": torch.tensor([[False], [True]]),
... "agents": {"reward": torch.zeros(n_envs, n_agents, 1)},
... },
... [n_envs],
... )
>>> transform = DoneTransform(
... in_keys=["done", "terminated"],
... reward_key=("agents", "reward"),
... )
>>> td = transform(td)
>>> td["agents", "done"].shape
torch.Size([2, 3, 1])
>>> bool((td["agents", "done"] == td["done"].unsqueeze(-1)).all())
True
```

As a collector post-processing transform the same keys are expanded
under `"next"`:

```
>>> collected = TensorDict(
... {
... "next": TensorDict(
... {
... "done": torch.tensor([[False], [True]]),
... "terminated": torch.tensor([[False], [True]]),
... "agents": {"reward": torch.zeros(n_envs, n_agents, 1)},
... },
... [n_envs],
... )
... },
... [n_envs],
... )
>>> collected = DoneTransform(
... reward_key=("agents", "reward"),
... done_keys=["done", "terminated"],
... )(collected)
>>> collected["next", "agents", "done"].shape
torch.Size([2, 3, 1])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#DoneTransform.forward)

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

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#DoneTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform