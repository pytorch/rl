# TrajCounter

*class*torchrl.envs.transforms.TrajCounter(*out_key: NestedKey = 'traj_count'*, ***, *repeats: int | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TrajCounter)

Global trajectory counter transform.

TrajCounter can be used to count the number of trajectories (i.e., the number of times reset is called) in any
TorchRL environment.
This transform will work within a single node across multiple processes (see note below).
A single transform can only count the trajectories associated with a single done state, but nested done states are
accepted as long as their prefix matches the prefix of the counter key.

Parameters:

**out_key** (*NestedKey**,**optional*) - The entry name of the trajectory counter. Defaults to `"traj_count"`.

Examples

```
>>> from torchrl.envs import GymEnv, StepCounter, TrajCounter
>>> env = GymEnv("Pendulum-v1").append_transform(StepCounter(6))
>>> env = env.append_transform(TrajCounter())
>>> r = env.rollout(18, break_when_any_done=False) # 18 // 6 = 3 trajectories
>>> r["next", "traj_count"]
tensor([[0],
 [0],
 [0],
 [0],
 [0],
 [0],
 [1],
 [1],
 [1],
 [1],
 [1],
 [1],
 [2],
 [2],
 [2],
 [2],
 [2],
 [2]])
```

Note

Sharing a trajectory counter among workers can be done in multiple ways, but it will usually involve wrapping the environment in a [`EnvCreator`](torchrl.envs.EnvCreator.html#torchrl.envs.EnvCreator). Not doing so may result in an error during serialization of the transform. The counter will be shared among the workers, meaning that at any point in time, it is guaranteed that there will not be two environments that will share the same trajectory count (and each (step-count, traj-count) pair will be unique).
Here are examples of valid ways of sharing a `TrajCounter` object between processes:

```
>>> # Option 1: Create the trajectory counter outside the environment.
>>> # This requires the counter to be cloned within the transformed env, as a single transform object cannot have two parents.
>>> t = TrajCounter()
>>> def make_env(max_steps=4, t=t):
... # See CountingEnv in torchrl.test.mocking_classes
... env = TransformedEnv(CountingEnv(max_steps=max_steps), t.clone())
... env.transform.transform_observation_spec(env.base_env.observation_spec)
... return env
>>> penv = ParallelEnv(
... 2,
... [EnvCreator(make_env, max_steps=4), EnvCreator(make_env, max_steps=5)],
... mp_start_method="spawn",
... )
>>> # Option 2: Create the transform within the constructor.
>>> # In this scenario, we still need to tell each sub-env what kwarg has to be used.
>>> # Both EnvCreator and ParallelEnv offer that possibility.
>>> def make_env(max_steps=4):
... t = TrajCounter()
... env = TransformedEnv(CountingEnv(max_steps=max_steps), t)
... env.transform.transform_observation_spec(env.base_env.observation_spec)
... return env
>>> make_env_c0 = EnvCreator(make_env)
>>> # Create a variant of the env with different kwargs
>>> make_env_c1 = make_env_c0.make_variant(max_steps=5)
>>> penv = ParallelEnv(
... 2,
... [make_env_c0, make_env_c1],
... mp_start_method="spawn",
... )
>>> # Alternatively, pass the kwargs to the ParallelEnv
>>> penv = ParallelEnv(
... 2,
... [make_env_c0, make_env_c0],
... create_env_kwargs=[{"max_steps": 5}, {"max_steps": 4}],
... mp_start_method="spawn",
... )
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TrajCounter.forward)

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

load_state_dict(*state_dict: Mapping[str, Any]*, *strict: bool = True*, *assign: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TrajCounter.load_state_dict)

Copy parameters and buffers from `state_dict` into this module and its descendants.

If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module's [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function.

Warning

If `assign` is `True` the optimizer must be created after
the call to `load_state_dict` unless
[`get_swap_module_params_on_conversion()`](https://docs.pytorch.org/docs/stable/future_mod.html#torch.__future__.get_swap_module_params_on_conversion) is `True`.

Parameters:

- **state_dict** (*dict*) - a dict containing parameters and
persistent buffers.
- **strict** (*bool**,**optional*) - whether to strictly enforce that the keys
in `state_dict` match the keys returned by this module's
[`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function. Default: `True`
- **assign** (*bool**,**optional*) - When set to `False`, the properties of the tensors
in the current module are preserved whereas setting it to `True` preserves
properties of the Tensors in the state dict. The only
exception is the `requires_grad` field of `Parameter`
for which the value from the module is preserved. Default: `False`

Returns:

- `missing_keys` is a list of str containing any keys that are expected

by this module but missing from the provided `state_dict`.
- `unexpected_keys` is a list of str containing the keys that are not

expected by this module but present in the provided `state_dict`.

Return type:

`NamedTuple` with `missing_keys` and `unexpected_keys` fields

Note

If a parameter or buffer is registered as `None` and its corresponding key
exists in `state_dict`, `load_state_dict()` will raise a
`RuntimeError`.

state_dict(**args*, *destination=None*, *prefix=''*, *keep_vars=False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TrajCounter.state_dict)

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to `None` are not included.

Note

The returned object is a shallow copy. It contains references
to the module's parameters and buffers.

Warning

Currently `state_dict()` also accepts positional arguments for
`destination`, `prefix` and `keep_vars` in order. However,
this is being deprecated and keyword arguments will be enforced in
future releases.

Warning

Please avoid the use of argument `destination` as it is not
designed for end-users.

Parameters:

- **destination** (*dict**,**optional*) - If provided, the state of module will
be updated into the dict and the same object is returned.
Otherwise, an `OrderedDict` will be created and returned.
Default: `None`.
- **prefix** (*str**,**optional*) - a prefix added to parameter and buffer
names to compose the keys in state_dict. Default: `''`.
- **keep_vars** (*bool**,**optional*) - by default the [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) s
returned in the state dict are detached from autograd. If it's
set to `True`, detaching will not be performed.
Default: `False`.

Returns:

a dictionary containing a whole state of the module

Return type:

dict

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight']
```

transform_observation_spec(*observation_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TrajCounter.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform