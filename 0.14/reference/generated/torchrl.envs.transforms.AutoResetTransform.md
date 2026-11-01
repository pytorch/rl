# AutoResetTransform

*class*torchrl.envs.transforms.AutoResetTransform(***, *replace: bool | None = None*, *fill_float='nan'*, *fill_int=-1*, *fill_bool=False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#AutoResetTransform)

A transform for auto-resetting environments.

This transform can be appended to any auto-resetting environment, or automatically
appended using `env = SomeEnvClass(..., auto_reset=True)`. If the transform is explicitly
appended to an env, a [`AutoResetEnv`](torchrl.envs.transforms.AutoResetEnv.html#torchrl.envs.transforms.AutoResetEnv) must be used.

An auto-reset environment must have the following properties (differences from this
description should be accounted for by subclassing this class):

> - the reset function can be called once at the beginning (after instantiation) with
> or without effect. Whether calls to reset are allowed after that is up to the
> environment itself.
> - During a rollout, any `done` state will result in a reset and produce an observation
> that isn't the last observation of the current episode, but the first observation
> of the next episode (this transform will extract and cache this observation
> and fill the obs with some arbitrary value).

Keyword Arguments:

- **replace** (*bool**,**optional*) - if `False`, values are just placed as they are in the
`"next"` entry even if they are not valid. Defaults to `True`. A value of
`False` overrides any subsequent filling keyword argument.
This argument can also be passed with the constructor method by passing a
`auto_reset_replace` argument: `env = FooEnv(..., auto_reset=True, auto_reset_replace=False)`.
- **fill_float** (`float` or str, optional) - The filling value for floating point tensors
that terminate an episode. A value of `None` means no replacement (values are just
placed as they are in the `"next"` entry even if they are not valid).
- **fill_int** (*int**,**optional*) - The filling value for signed integer tensors
that terminate an episode. A value of `None` means no replacement (values are just
placed as they are in the `"next"` entry even if they are not valid).
- **fill_bool** (*bool**,**optional*) - The filling value for boolean tensors
that terminate an episode. A value of `None` means no replacement (values are just
placed as they are in the `"next"` entry even if they are not valid).

Arguments are only available when the transform is explicitly instantiated (not through EnvType(..., auto_reset=True)).

Examples

```
>>> from torchrl.envs import GymEnv
>>> from torchrl.envs import set_gym_backend
>>> import torch
>>> torch.manual_seed(0)
>>>
>>> class AutoResettingGymEnv(GymEnv):
... def _step(self, tensordict):
... tensordict = super()._step(tensordict)
... if tensordict["done"].any():
... td_reset = super().reset()
... tensordict.update(td_reset.exclude(*self.done_keys))
... return tensordict
...
... def _reset(self, tensordict=None):
... if tensordict is not None and "_reset" in tensordict:
... return tensordict.copy()
... return super()._reset(tensordict)
>>>
>>> with set_gym_backend("gym"):
... env = AutoResettingGymEnv("CartPole-v1", auto_reset=True, auto_reset_replace=True)
... env.set_seed(0)
... r = env.rollout(30, break_when_any_done=False)
>>> print(r["next", "done"].squeeze())
tensor([False, False, False, False, False, False, False, False, False, False,
 False, False, False, True, False, False, False, False, False, False,
 False, False, False, False, False, True, False, False, False, False])
>>> print("observation after reset are set as nan", r["next", "observation"])
observation after reset are set as nan tensor([[-4.3633e-02, -1.4877e-01, 1.2849e-02, 2.7584e-01],
 [-4.6609e-02, 4.6166e-02, 1.8366e-02, -1.2761e-02],
 [-4.5685e-02, 2.4102e-01, 1.8111e-02, -2.9959e-01],
 [-4.0865e-02, 4.5644e-02, 1.2119e-02, -1.2542e-03],
 [-3.9952e-02, 2.4059e-01, 1.2094e-02, -2.9009e-01],
 [-3.5140e-02, 4.3554e-01, 6.2920e-03, -5.7893e-01],
 [-2.6429e-02, 6.3057e-01, -5.2867e-03, -8.6963e-01],
 [-1.3818e-02, 8.2576e-01, -2.2679e-02, -1.1640e+00],
 [ 2.6972e-03, 1.0212e+00, -4.5959e-02, -1.4637e+00],
 [ 2.3121e-02, 1.2168e+00, -7.5232e-02, -1.7704e+00],
 [ 4.7457e-02, 1.4127e+00, -1.1064e-01, -2.0854e+00],
 [ 7.5712e-02, 1.2189e+00, -1.5235e-01, -1.8289e+00],
 [ 1.0009e-01, 1.0257e+00, -1.8893e-01, -1.5872e+00],
 [ nan, nan, nan, nan],
 [-3.9405e-02, -1.7766e-01, -1.0403e-02, 3.0626e-01],
 [-4.2959e-02, -3.7263e-01, -4.2775e-03, 5.9564e-01],
 [-5.0411e-02, -5.6769e-01, 7.6354e-03, 8.8698e-01],
 [-6.1765e-02, -7.6292e-01, 2.5375e-02, 1.1820e+00],
 [-7.7023e-02, -9.5836e-01, 4.9016e-02, 1.4826e+00],
 [-9.6191e-02, -7.6387e-01, 7.8667e-02, 1.2056e+00],
 [-1.1147e-01, -9.5991e-01, 1.0278e-01, 1.5219e+00],
 [-1.3067e-01, -7.6617e-01, 1.3322e-01, 1.2629e+00],
 [-1.4599e-01, -5.7298e-01, 1.5848e-01, 1.0148e+00],
 [-1.5745e-01, -7.6982e-01, 1.7877e-01, 1.3527e+00],
 [-1.7285e-01, -9.6668e-01, 2.0583e-01, 1.6956e+00],
 [ nan, nan, nan, nan],
 [-4.3962e-02, 1.9845e-01, -4.5015e-02, -2.5903e-01],
 [-3.9993e-02, 3.9418e-01, -5.0196e-02, -5.6557e-01],
 [-3.2109e-02, 5.8997e-01, -6.1507e-02, -8.7363e-01],
 [-2.0310e-02, 3.9574e-01, -7.8980e-02, -6.0090e-01]])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#AutoResetTransform.forward)

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