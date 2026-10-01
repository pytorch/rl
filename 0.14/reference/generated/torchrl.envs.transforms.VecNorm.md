# VecNorm

*class*torchrl.envs.transforms.VecNorm(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm)

Moving average normalization layer for torchrl environments.

Warning

This class is to be deprecated in favor of `VecNormV2` and will be replaced by
that class in v0.10. You can adapt to these changes by using the new_api argument or importing the
VecNormV2 class from torchrl.envs.

VecNorm keeps track of the summary statistics of a dataset to standardize
it on-the-fly. If the transform is in 'eval' mode, the running
statistics are not updated.

If multiple processes are running a similar environment, one can pass a
TensorDictBase instance that is placed in shared memory: if so, every time
the normalization layer is queried it will update the values for all
processes that share the same reference.

To use VecNorm at inference time and avoid updating the values with the new
observations, one should substitute this layer by `to_observation_norm()`.
This will provide a static version of VecNorm which will not be updated
when the source transform is updated.
To get a frozen copy of the VecNorm layer, see `frozen_copy()`.

Parameters:

- **in_keys** (*sequence**of**NestedKey**,**optional*) - keys to be updated.
default: ["observation", "reward"]
- **out_keys** (*sequence**of**NestedKey**,**optional*) - destination keys.
Defaults to `in_keys`.
- **shared_td** (*TensorDictBase**,**optional*) - A shared tensordict containing the
keys of the transform.
- **lock** (*mp.Lock*) - a lock to prevent race conditions between processes.
Defaults to None (lock created during init).
- **decay** (*number**,**optional*) - decay rate of the moving average.
default: 0.99
- **eps** (*number**,**optional*) - lower bound of the running standard
deviation (for numerical underflow). Default is 1e-4.
- **shapes** (*List**[*[*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*]**,**optional*) - if provided, represents the shape
of each in_keys. Its length must match the one of `in_keys`.
Each shape must match the trailing dimension of the corresponding
entry.
If not, the feature dimensions of the entry (ie all dims that do
not belong to the tensordict batch-size) will be considered as
feature dimension.
- **new_api** (*bool**or**None**,**optional*) - if `True`, an instance of VecNormV2 will be returned.
If not passed, a warning will be raised.
Defaults to `False`.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> t = VecNorm(decay=0.9)
>>> env = GymEnv("Pendulum-v0")
>>> env = TransformedEnv(env, t)
>>> tds = []
>>> for _ in range(1000):
... td = env.rand_step()
... if td.get("done"):
... _ = env.reset()
... tds += [td]
>>> tds = torch.stack(tds, 0)
>>> print((abs(tds.get(("next", "observation")).mean(0))<0.2).all())
tensor(True)
>>> print((abs(tds.get(("next", "observation")).std(0)-1)<0.2).all())
tensor(True)
```

To recover the original (denormalized) values from normalized data, use `denorm()`:

```
>>> denormed = t.denorm(tds)
```

*static*build_td_for_shared_vecnorm(*env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*, *keys: Sequence[str] | None = None*, *memmap: bool = False*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.build_td_for_shared_vecnorm)

Creates a shared tensordict for normalization across processes.

Parameters:

- **env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - example environment to be used to create the
tensordict
- **keys** (*sequence**of**NestedKey**,**optional*) - keys that
have to be normalized. Default is ["next", "reward"]
- **memmap** (*bool*) - if `True`, the resulting tensordict will be cast into
memory map (using memmap_()). Otherwise, the tensordict
will be placed in shared memory.

Returns:

A memory in shared memory to be sent to each process.

Examples

```
>>> from torch import multiprocessing as mp
>>> queue = mp.Queue()
>>> env = make_env()
>>> td_shared = VecNorm.build_td_for_shared_vecnorm(env,
... ["next", "reward"])
>>> assert td_shared.is_shared()
>>> queue.put(td_shared)
>>> # on workers
>>> v = VecNorm(shared_td=queue.get())
>>> env = TransformedEnv(make_env(), v)
```

denorm(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.denorm)

Denormalize a tensordict using the inverse of the normalization transform.

Applies the inverse of the normalization: `original = normalized * scale + loc`.

Reads normalized values from `out_keys` and writes denormalized values to `in_keys`.

Parameters:

**tensordict** (*TensorDictBase*) - the tensordict containing normalized values.

Returns:

A shallow copy of the tensordict with denormalized values written to `in_keys`.

Raises:

**RuntimeError** - if the transform has not been initialized (no data seen yet).

Examples

```
>>> from torchrl.envs import GymEnv, VecNorm
>>> env = GymEnv("Pendulum-v1")
>>> vecnorm = VecNorm(in_keys=["observation"], out_keys=["observation_norm"])
>>> env = env.append_transform(vecnorm)
>>> # Collect some data to initialize statistics
>>> rollout = env.rollout(10)
>>> # Denormalize the normalized observations
>>> denormed = vecnorm.denorm(rollout)
>>> # denormed["observation"] now contains the original scale values
```

forward(*next_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

freeze() → VecNorm[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.freeze)

Freezes the VecNorm, avoiding the stats to be updated when called.

See `unfreeze()`.

frozen_copy() → VecNorm[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.frozen_copy)

Returns a copy of the Transform that keeps track of the stats but does not update them.

get_extra_state() → OrderedDict[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.get_extra_state)

Return any extra state to include in the module's state_dict.

Implement this and a corresponding `set_extra_state()` for your module
if you need to store extra state. This function is called when building the
module's state_dict().

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:

Any extra state to store in the module's state_dict

Return type:

object

*property*loc

Returns a TensorDict with the loc to be used for an affine transform.

*property*scale

Returns a TensorDict with the scale to be used for an affine transform.

set_extra_state(*state: OrderedDict*) → None[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.set_extra_state)

Set extra state contained in the loaded state_dict.

This function is called from `load_state_dict()` to handle any extra state
found within the state_dict. Implement this function and a corresponding
`get_extra_state()` for your module if you need to store extra state within its
state_dict.

Parameters:

**state** (*dict*) - Extra state from the state_dict

*property*standard_normal*: bool*

Whether the affine transform given by loc and scale follows the standard normal equation.

Similar to `ObservationNorm` standard_normal attribute.

Always returns `True`.

to_observation_norm() → [Compose](torchrl.envs.transforms.Compose.html#torchrl.envs.transforms.Compose) | [ObservationNorm](torchrl.envs.transforms.ObservationNorm.html#torchrl.envs.transforms.ObservationNorm)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.to_observation_norm)

Converts VecNorm into an ObservationNorm class that can be used at inference time.

The `ObservationNorm` layer can be updated using the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict)
API.

Examples

```
>>> from torchrl.envs import GymEnv, VecNorm
>>> vecnorm = VecNorm(in_keys=["observation"])
>>> train_env = GymEnv("CartPole-v1", device=None).append_transform(
... vecnorm)
>>>
>>> r = train_env.rollout(4)
>>>
>>> eval_env = GymEnv("CartPole-v1").append_transform(
... vecnorm.to_observation_norm())
>>> print(eval_env.transform.loc, eval_env.transform.scale)
>>>
>>> r = train_env.rollout(4)
>>> # Update entries with state_dict
>>> eval_env.transform.load_state_dict(
... vecnorm.to_observation_norm().state_dict())
>>> print(eval_env.transform.loc, eval_env.transform.scale)
```

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

unfreeze() → VecNorm[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#VecNorm.unfreeze)

Unfreezes the VecNorm.

See `freeze()`.