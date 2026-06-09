# VecGymEnvTransform

*class*torchrl.envs.transforms.VecGymEnvTransform(*final_name: str = 'final'*, *missing_obs_value: Any = nan*, ***, *native_autoreset: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#VecGymEnvTransform)

A transform for GymWrapper subclasses that handles the auto-reset in a consistent way.

Gym, gymnasium and SB3 provide vectorized (read, parallel or batched) environments
that are automatically reset. When this occurs, the actual observation resulting
from the action is saved within a key in the info.
The class `torchrl.envs.libs.gym.terminal_obs_reader` reads that observation
and stores it in a `"final"` key within the output tensordict.
In turn, this transform reads that final data, swaps it with the observation
written in its place that results from the actual reset, and saves the
reset output in a private container. The resulting data truly reflects
the output of the step.

This class works from gym 0.13 till the most recent gymnasium version.

Note

Gym versions < 0.22 did not return the final observations. For these,
we simply fill the next observations with NaN (because it is lost) and
do the swap at the next step.

Then, when calling env.reset, the saved data is written back where it belongs
(and the reset is a no-op).

This transform is automatically appended to the gym env whenever the wrapper
is created with an async env.

Parameters:

- **final_name** (*str**,**optional*) - the name of the final observation in the dict.
Defaults to "final".
- **missing_obs_value** (*Any**,**optional*) - default value to use as placeholder for missing
last observations. Defaults to np.nan.
- **native_autoreset** (*bool**,**optional*) - if `True`, leaves the native
auto-reset observation available to the environment wrapper so it
can be cloned into the next root observation, while the terminal
floating point `"next"` observation is marked with `NaN`.
Defaults to `False`.

Note

In general, this class should not be handled directly. It is
created whenever a vectorized environment is placed within a `GymWrapper`.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#VecGymEnvTransform.forward)

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

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#VecGymEnvTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform