# TanhModule

*class*torchrl.modules.tensordict_module.TanhModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#TanhModule)

A Tanh module for deterministic policies with bounded action space.

This transform is to be used as a TensorDictModule layer to map a network
output to a bounded space.

Parameters:

- **in_keys** (*list**of**str**or**tuples**of**str*) - the input keys of the module.
- **out_keys** (*list**of**str**or**tuples**of**str**,**optional*) - the output keys of the module.
If none is provided, the same keys as in_keys are assumed.

Keyword Arguments:

- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - if provided, the spec of the output.
If a Composite is provided, its key(s) must match the key(s)
in out_keys. Otherwise, the key(s) of out_keys are assumed and the
same spec is used for all outputs.
- **low** (`float`, np.ndarray or torch.Tensor) - the lower bound of the space.
If none is provided and no spec is provided, -1 is assumed. If a
spec is provided, the minimum value of the spec will be retrieved.
- **high** (`float`, np.ndarray or torch.Tensor) - the higher bound of the space.
If none is provided and no spec is provided, 1 is assumed. If a
spec is provided, the maximum value of the spec will be retrieved.
- **clamp** (*bool**,**optional*) - if `True`, the outputs will be clamped to be
within the boundaries but at a minimum resolution from them.
Defaults to `False`.

Examples

```
>>> from tensordict import TensorDict
>>> # simplest use case: -1 - 1 boundaries
>>> torch.manual_seed(0)
>>> in_keys = ["action"]
>>> mod = TanhModule(
... in_keys=in_keys,
... )
>>> data = TensorDict({"action": torch.randn(5) * 10}, [])
>>> data = mod(data)
>>> data['action']
tensor([ 1.0000, -0.9944, -1.0000, 1.0000, -1.0000])
>>> # low and high can be customized
>>> low = -2
>>> high = 1
>>> mod = TanhModule(
... in_keys=in_keys,
... low=low,
... high=high,
... )
>>> data = TensorDict({"action": torch.randn(5) * 10}, [])
>>> data = mod(data)
>>> data['action']
tensor([-2.0000, 0.9991, 1.0000, -2.0000, -1.9991])
>>> # A spec can be provided
>>> from torchrl.data import Bounded
>>> spec = Bounded(low, high, shape=())
>>> mod = TanhModule(
... in_keys=in_keys,
... low=low,
... high=high,
... spec=spec,
... clamp=False,
... )
>>> # One can also work with multiple keys
>>> in_keys = ['a', 'b']
>>> spec = Composite(
... a=Bounded(-3, 0, shape=()),
... b=Bounded(0, 3, shape=()))
>>> mod = TanhModule(
... in_keys=in_keys,
... spec=spec,
... )
>>> data = TensorDict(
... {'a': torch.randn(10), 'b': torch.randn(10)}, batch_size=[])
>>> data = mod(data)
>>> data['a']
tensor([-2.3020, -1.2299, -2.5418, -0.2989, -2.6849, -1.3169, -2.2690, -0.9649,
 -2.5686, -2.8602])
>>> data['b']
tensor([2.0315, 2.8455, 2.6027, 2.4746, 1.7843, 2.7782, 0.2111, 0.5115, 1.4687,
 0.5760])
```

forward(*tensordict=None*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#TanhModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.