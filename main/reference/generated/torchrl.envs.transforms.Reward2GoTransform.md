# Reward2GoTransform

*class*torchrl.envs.transforms.Reward2GoTransform(*gamma: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = 1.0*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *done_key: NestedKey | None = 'done'*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#Reward2GoTransform)

Calculates the reward to go based on the episode reward and a discount factor.

As the `Reward2GoTransform` is only an inverse transform the `in_keys` will be directly used for the `in_keys_inv`.
The reward-to-go can be only calculated once the episode is finished. Therefore, the transform should be applied to the replay buffer
and not to the collector or within an environment.

Parameters:

- **gamma** (`float` or torch.Tensor) - the discount factor. Defaults to 1.0.
- **in_keys** (*sequence**of**NestedKey*) - the entries to rename. Defaults to
`("next", "reward")` if none is provided.
- **out_keys** (*sequence**of**NestedKey*) - the entries to rename. Defaults to
the values of `in_keys` if none is provided.
- **done_key** (*NestedKey*) - the done entry. Defaults to `"done"`.
- **truncated_key** (*NestedKey*) - the truncated entry. Defaults to `"truncated"`.
If no truncated entry is found, only the `"done"` will be used.

Examples

```
>>> # Using this transform as part of a replay buffer
>>> from torchrl.data import ReplayBuffer, LazyTensorStorage
>>> torch.manual_seed(0)
>>> r2g = Reward2GoTransform(gamma=0.99, out_keys=["reward_to_go"])
>>> rb = ReplayBuffer(storage=LazyTensorStorage(100), transform=r2g)
>>> batch, timesteps = 4, 5
>>> done = torch.zeros(batch, timesteps, 1, dtype=torch.bool)
>>> for i in range(batch):
... while not done[i].any():
... done[i] = done[i].bernoulli_(0.1)
>>> reward = torch.ones(batch, timesteps, 1)
>>> td = TensorDict(
... {"next": {"done": done, "reward": reward}},
... [batch, timesteps],
... )
>>> rb.extend(td)
>>> sample = rb.sample(1)
>>> print(sample["next", "reward"])
tensor([[[1.],
 [1.],
 [1.],
 [1.],
 [1.]]])
>>> print(sample["reward_to_go"])
tensor([[[4.9010],
 [3.9404],
 [2.9701],
 [1.9900],
 [1.0000]]])
```

One can also use this transform directly with a collector: make sure to
append the inv method of the transform.

Examples

```
>>> from torchrl.modules import RandomPolicy >>> >>> >>> from torchrl.collectors import Collector
>>> from torchrl.envs.libs.gym import GymEnv
>>> t = Reward2GoTransform(gamma=0.99, out_keys=["reward_to_go"])
>>> env = GymEnv("Pendulum-v1")
>>> collector = Collector(
... env,
... RandomPolicy(env.action_spec),
... frames_per_batch=200,
... total_frames=-1,
... postproc=t.inv
... )
>>> for data in collector:
... break
>>> print(data)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward_to_go: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False)
```

Using this transform as part of an env will raise an exception

Examples

```
>>> t = Reward2GoTransform(gamma=0.99)
>>> TransformedEnv(GymEnv("Pendulum-v1"), t) # crashes
```

Note

In settings where multiple done entries are present, one should build
a single `Reward2GoTransform` for each done-reward pair.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#Reward2GoTransform.forward)

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