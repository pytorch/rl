# VecNormV2

*class*torchrl.envs.transforms.VecNormV2(*in_keys: Sequence[NestedKey]*, *out_keys: Sequence[NestedKey] | None = None*, ***, *lock: Lock = None*, *stateful: bool = True*, *decay: float = 0.9999*, *eps: float = 0.0001*, *shared_data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *reduce_batch_dims: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2)

A class for normalizing vectorized observations and rewards in reinforcement learning environments.

VecNormV2 can operate in either a stateful or stateless mode. In stateful mode, it maintains
internal statistics (mean and variance) to normalize inputs. In stateless mode, it requires
external statistics to be provided for normalization.

Note

This class is designed to be an almost drop-in replacement for [`VecNorm`](torchrl.envs.transforms.VecNorm.html#torchrl.envs.transforms.VecNorm).
It should not be constructed directly, but rather with the [`VecNorm`](torchrl.envs.transforms.VecNorm.html#torchrl.envs.transforms.VecNorm)
transform using the new_api=True keyword argument. In v0.10, the [`VecNorm`](torchrl.envs.transforms.VecNorm.html#torchrl.envs.transforms.VecNorm)
transform will be switched to the new api by default.

Stateful vs. Stateless:

Stateful Mode (stateful=True):

> - Maintains internal statistics (loc, var, count) for normalization.
> - Updates statistics with each call unless frozen.
> - state_dict returns the current statistics.
> - load_state_dict updates the internal statistics with the provided state.

Stateless Mode (stateful=False):

> - Requires external statistics to be provided for normalization.
> - Does not maintain or update internal statistics.
> - state_dict returns an empty dictionary.
> - load_state_dict does not affect internal state.

Parameters:

- **in_keys** (*Sequence**[**NestedKey**]*) - The input keys for the data to be normalized.
- **out_keys** (*Sequence**[**NestedKey**]**|**None*) - The output keys for the normalized data. Defaults to in_keys if
not provided.
- **lock** (*mp.Lock**,**optional*) - A lock for thread safety.
- **stateful** (*bool**,**optional*) - Whether the VecNorm is stateful. Stateless versions of this
transform requires the data to be carried within the input/output tensordicts.
Defaults to True.
- **decay** (*float**,**optional*) - The decay rate for updating statistics. Defaults to 0.9999.
If decay=1 is used, the normalizing statistics have an infinite memory (each item is weighed
identically). Lower values weigh recent data more than old ones.
- **eps** (*float**,**optional*) - A small value to prevent division by zero. Defaults to 1e-4.
- **shared_data** (*TensorDictBase**|**None**,**optional*) - Shared data for initialization. Defaults to None.
- **reduce_batch_dims** (*bool**,**optional*) - If True, the batch dimensions are reduced by averaging the data
before updating the statistics. This is useful when samples are received in batches, as it allows
the moving average to be computed over the entire batch rather than individual elements. Note that
this option is only supported in stateful mode (stateful=True). Defaults to False.

Variables:

- **stateful** (*bool*) - Indicates whether the VecNormV2 is stateful or stateless.
- **lock** (*mp.Lock*) - A multiprocessing lock to ensure thread safety when updating statistics.
- **decay** (*float*) - The decay rate for updating statistics.
- **eps** (*float*) - A small value to prevent division by zero during normalization.
- **frozen** (*bool*) - Indicates whether the VecNormV2 is frozen, preventing updates to statistics.
- **_cast_int_to_float** (*bool*) - Indicates whether integer inputs should be cast to float.

freeze()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.freeze)

Freezes the VecNorm, preventing updates to statistics.

unfreeze()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.unfreeze)

Unfreezes the VecNorm, allowing updates to statistics.

frozen_copy()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.frozen_copy)

Returns a frozen copy of the VecNorm.

clone()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.clone)

Returns a clone of the VecNorm.

denorm(*tensordict*)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.denorm)

Denormalizes data using the inverse of the normalization (stateful mode only).

transform_observation_spec(*observation_spec*)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.transform_observation_spec)

Transforms the observation specification.

transform_reward_spec(*reward_spec*, *observation_spec*)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.transform_reward_spec)

Transforms the reward specification.

transform_output_spec(*output_spec*)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.transform_output_spec)

Transforms the output specification.

to_observation_norm()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.to_observation_norm)

Converts the VecNorm to an ObservationNorm transform.

set_extra_state(*state*)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.set_extra_state)

Sets the extra state for the VecNorm.

get_extra_state()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.get_extra_state)

Gets the extra state of the VecNorm.

loc()

Returns the location (mean) for normalization.

scale()

Returns the scale (standard deviation) for normalization.

standard_normal()

Indicates whether the normalization follows the standard normal distribution.

State Dict Behavior:

> - In stateful mode, state_dict returns a dictionary containing the current loc, var, and count.
> These can be used to share the tensors across processes (this method is automatically triggered by
> `VecNorm` to share the VecNorm states across processes).
> - In stateless mode, state_dict returns an empty dictionary as no internal state is maintained.

Load State Dict Behavior:

> - In stateful mode, load_state_dict updates the internal loc, var, and count with the provided state.
> - In stateless mode, load_state_dict does not modify any internal state as there is none to update.

See also

[`VecNorm`](torchrl.envs.transforms.VecNorm.html#torchrl.envs.transforms.VecNorm) for the first version of this transform.

Examples

```
>>> import torch
>>> from torchrl.envs import EnvCreator, GymEnv, ParallelEnv, SerialEnv, VecNormV2
>>>
>>> torch.manual_seed(0)
>>> env = GymEnv("Pendulum-v1")
>>> env_trsf = env.append_transform(
>>> VecNormV2(in_keys=["observation", "reward"], out_keys=["observation_norm", "reward_norm"])
>>> )
>>> r = env_trsf.rollout(10)
>>> print("Unnormalized rewards", r["next", "reward"])
Unnormalized rewards tensor([[ -1.7967],
 [ -2.1238],
 [ -2.5911],
 [ -3.5275],
 [ -4.8585],
 [ -6.5028],
 [ -8.2505],
 [-10.3169],
 [-12.1332],
 [-13.1235]])
>>> print("Normalized rewards", r["next", "reward_norm"])
Normalized rewards tensor([[-1.6596e-04],
 [-8.3072e-02],
 [-1.9170e-01],
 [-3.9255e-01],
 [-5.9131e-01],
 [-7.4671e-01],
 [-8.3760e-01],
 [-9.2058e-01],
 [-9.3484e-01],
 [-8.6185e-01]])
>>> # Aggregate values when using batched envs
>>> env = SerialEnv(2, [lambda: GymEnv("Pendulum-v1")] * 2)
>>> env_trsf = env.append_transform(
>>> VecNormV2(
>>> in_keys=["observation", "reward"],
>>> out_keys=["observation_norm", "reward_norm"],
>>> # Use reduce_batch_dims=True to aggregate values across batch elements
>>> reduce_batch_dims=True, )
>>> )
>>> r = env_trsf.rollout(10)
>>> print("Unnormalized rewards", r["next", "reward"])
Unnormalized rewards tensor([[[-0.1456],
 [-0.1862],
 [-0.2053],
 [-0.2605],
 [-0.4046],
 [-0.5185],
 [-0.8023],
 [-1.1364],
 [-1.6183],
 [-2.5406]],
```

> [[-0.0920],
> 
> [-0.1492],
> [-0.2702],
> [-0.3917],
> [-0.5001],
> [-0.7947],
> [-1.0160],
> [-1.3347],
> [-1.9082],
> [-2.9679]]])

```
>>> print("Normalized rewards", r["next", "reward_norm"])
Normalized rewards tensor([[[-0.2199],
 [-0.2918],
 [-0.1668],
 [-0.2083],
 [-0.4981],
 [-0.5046],
 [-0.7950],
 [-0.9791],
 [-1.1484],
 [-1.4182]],
```

> [[ 0.2201],
> 
> [-0.0403],
> [-0.5206],
> [-0.7791],
> [-0.8282],
> [-1.2306],
> [-1.2279],
> [-1.2907],
> [-1.4929],
> [-1.7793]]])

```
>>> print("Loc / scale", env_trsf.transform.loc["reward"], env_trsf.transform.scale["reward"])
Loc / scale tensor([-0.8626]) tensor([1.1832])
>>>
>>> # Share values between workers
>>> def make_env():
... env = GymEnv("Pendulum-v1")
... env_trsf = env.append_transform(
... VecNormV2(in_keys=["observation", "reward"], out_keys=["observation_norm", "reward_norm"])
... )
... return env_trsf
...
...
>>> if __name__ == "__main__":
... # EnvCreator will share the loc/scale vals
... make_env = EnvCreator(make_env)
... # Create a local env to track the loc/scale
... local_env = make_env()
... env = ParallelEnv(2, [make_env] * 2)
... r = env.rollout(10)
... # Non-zero loc and scale testify that the sub-envs share their summary stats with us
... print("Remotely updated loc / scale", local_env.transform.loc["reward"], local_env.transform.scale["reward"])
Remotely updated loc / scale tensor([-0.4307]) tensor([0.9613])
... env.close()
```

To recover the original (denormalized) values from normalized data, use `denorm()`:

```
>>> denormed = env_trsf.transform.denorm(r)
```

Note

The `denorm()` method is only available in stateful mode.

denorm(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.denorm)

Denormalize a tensordict using the inverse of the normalization transform.

Applies the inverse of the normalization: `original = normalized * scale + loc`.

Reads normalized values from `out_keys` and writes denormalized values to `in_keys`.

Note

This method is only available in stateful mode.

Parameters:

**tensordict** (*TensorDictBase*) - the tensordict containing normalized values.

Returns:

A shallow copy of the tensordict with denormalized values written to `in_keys`.

Raises:

- **NotImplementedError** - if the transform is in stateless mode.
- **RuntimeError** - if the transform has not been initialized (no data seen yet).

Examples

```
>>> from torchrl.envs import GymEnv
>>> from torchrl.envs.transforms import VecNormV2
>>> env = GymEnv("Pendulum-v1")
>>> vecnorm = VecNormV2(
... in_keys=["observation"],
... out_keys=["observation_norm"],
... stateful=True,
... )
>>> env = env.append_transform(vecnorm)
>>> # Collect some data to initialize statistics
>>> rollout = env.rollout(10)
>>> # Denormalize the normalized observations
>>> denormed = vecnorm.denorm(rollout)
>>> # denormed["observation"] now contains the original scale values
```

freeze() → VecNormV2[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.freeze)

Freezes the VecNorm, avoiding the stats to be updated when called.

See `unfreeze()`.

frozen_copy()[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.frozen_copy)

Returns a copy of the Transform that keeps track of the stats but does not update them.

get_extra_state() → OrderedDict[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.get_extra_state)

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

set_extra_state(*state: OrderedDict*) → None[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.set_extra_state)

Set extra state contained in the loaded state_dict.

This function is called from `load_state_dict()` to handle any extra state
found within the state_dict. Implement this function and a corresponding
`get_extra_state()` for your module if you need to store extra state within its
state_dict.

Parameters:

**state** (*dict*) - Extra state from the state_dict

*property*standard_normal

Whether the affine transform given by loc and scale follows the standard normal equation.

Similar to `ObservationNorm` standard_normal attribute.

Always returns `True`.

transform_observation_spec(*observation_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*, *observation_spec*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

unfreeze() → VecNormV2[[source]](../../_modules/torchrl/envs/transforms/vecnorm.html#VecNormV2.unfreeze)

Unfreezes the VecNorm.

See `freeze()`.