# ConsistentDropoutModule

*class*torchrl.modules.ConsistentDropoutModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/models/exploration.html#ConsistentDropoutModule)

A TensorDictModule wrapper for `ConsistentDropout`.

Parameters:

- **p** (`float`, optional) - Dropout probability. Default: `0.5`.
- **in_keys** (*NestedKey**or**list**of**NestedKeys*) - keys to be read
from input tensordict and passed to this module.
- **out_keys** (*NestedKey**or**iterable**of**NestedKeys*) - keys to be written to the input tensordict.
Defaults to `in_keys` values.

Keyword Arguments:

- **input_shape** (*tuple**,**optional*) - the shape of the input (non-batchted), used to generate the
tensordict primers with `make_tensordict_primer()`.
- **input_dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the input for the primer. If none is passed,
`torch.get_default_dtype` is assumed.

Note

To use this class within a policy, one needs the mask to be reset at reset time.
This can be achieved through a `TensorDictPrimer` transform that can be obtained
with `make_tensordict_primer()`. See this method for more information.

Examples

```
>>> from tensordict import TensorDict
>>> module = ConsistentDropoutModule(p = 0.1)
>>> td = TensorDict({"x": torch.randn(3, 4)}, [3])
>>> module(td)
TensorDict(
 fields={
 mask_6127171760: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
 x: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

forward(*tensordict*)[[source]](../../_modules/torchrl/modules/models/exploration.html#ConsistentDropoutModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

make_tensordict_primer()[[source]](../../_modules/torchrl/modules/models/exploration.html#ConsistentDropoutModule.make_tensordict_primer)

Makes a tensordict primer for the environment to generate random masks during reset calls.

See also

`torchrl.modules.utils.get_primers_from_module()` for a method to generate all primers for a given

module.

Examples

```
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>> from torchrl.envs import GymEnv, StepCounter, SerialEnv
>>> m = Seq(
... Mod(torch.nn.Linear(7, 4), in_keys=["observation"], out_keys=["intermediate"]),
... ConsistentDropoutModule(
... p=0.5,
... input_shape=(2, 4),
... in_keys="intermediate",
... ),
... Mod(torch.nn.Linear(4, 7), in_keys=["intermediate"], out_keys=["action"]),
... )
>>> primer = get_primers_from_module(m)
>>> env0 = GymEnv("Pendulum-v1").append_transform(StepCounter(5))
>>> env1 = GymEnv("Pendulum-v1").append_transform(StepCounter(6))
>>> env = SerialEnv(2, [lambda env=env0: env, lambda env=env1: env])
>>> env = env.append_transform(primer)
>>> r = env.rollout(10, m, break_when_any_done=False)
>>> mask = [k for k in r.keys() if k.startswith("mask")][0]
>>> assert (r[mask][0, :5] != r[mask][0, 5:6]).any()
>>> assert (r[mask][0, :4] == r[mask][0, 4:5]).all()
```