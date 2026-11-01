# SafeProbabilisticModule

*class*torchrl.modules.tensordict_module.SafeProbabilisticModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/probabilistic.html#SafeProbabilisticModule)

[`tensordict.nn.ProbabilisticTensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) subclass that accepts a `TensorSpec` as an argument to control the output domain.

SafeProbabilisticModule is a non-parametric module embedding a
probability distribution constructor. It reads the distribution parameters from an input
TensorDict using the specified in_keys and outputs a sample (loosely speaking) of the
distribution.

The output "sample" is produced given some rule, specified by the input `default_interaction_type`
argument and the `interaction_type()` global function.

SafeProbabilisticModule can be used to construct the distribution
(through the `get_dist()` method) and/or sampling from this distribution
(through a regular `__call__()` to the module).

A SafeProbabilisticModule instance has two main features:

- It reads and writes from and to TensorDict objects;
- It uses a real mapping R^n -> R^m to create a distribution in R^d from
which values can be sampled or computed.

When the `__call__` and `forward` methods are called, a distribution is
created, and a value computed (depending on the `interaction_type` value, 'dist.mean',
'dist.mode', 'dist.median' attributes could be used, as well as
the 'dist.rsample', 'dist.sample' method). The sampling step is skipped if the supplied
TensorDict has all the desired key-value pairs already.

By default, SafeProbabilisticModule distribution class is a `Delta`
distribution, making SafeProbabilisticModule a simple wrapper around
a deterministic mapping function.

This class differs from [`tensordict.nn.ProbabilisticTensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) in that it accepts a `spec`
keyword argument which can be used to control whether samples belong to the distribution or not. The `safe`
keyword argument controls whether the samples values should be checked against the spec.

Parameters:

- **in_keys** (*NestedKey**|**List**[**NestedKey**]**|**Dict**[**str**,**NestedKey**]*) - key(s) that will be read from the input TensorDict
and used to build the distribution.
Importantly, if it's a list of NestedKey or a NestedKey, the leaf (last element) of those keys must match the keywords used by
the distribution class of interest, e.g. `"loc"` and `"scale"` for
the `Normal` distribution and similar.
If in_keys is a dictionary, the keys are the keys of the distribution and the values are the keys in the
tensordict that will get match to the corresponding distribution keys.
- **out_keys** (*NestedKey**|**List**[**NestedKey**]**|**None*) - key(s) where the sampled values will be written.
Importantly, if these keys are found in the input TensorDict, the sampling step will be skipped.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - specs of the first output tensor. Used when calling
td_module.random() to generate random values in the target space.

Keyword Arguments:

- **safe** (*bool**,**optional*) - if `True`, the value of the sample is checked against the
input spec. Out-of-domain sampling can occur because of exploration policies
or numerical under/overflow issues. As for the `spec` argument, this
check will only occur for the distribution sample, but not the other tensors
returned by the input module. If the sample is out of bounds, it is
projected back onto the desired space using the TensorSpec.project method.
Default is `False`.
- **default_interaction_type** (*InteractionType**,**optional*) -

keyword-only argument.
Default method to be used to retrieve
the output value. Should be one of InteractionType: MODE, MEDIAN, MEAN or RANDOM
(in which case the value is sampled randomly from the distribution). Default
is MODE.

Note

When a sample is drawn, the
`ProbabilisticTensorDictModule` instance will
first look for the interaction mode dictated by the
`interaction_type()`
global function. If this returns None (its default value), then the
default_interaction_type of the ProbabilisticTDModule
instance will be used. Note that
[`BaseCollector`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)
instances will use set_interaction_type to
`tensordict.nn.InteractionType.RANDOM` by default.

Note

In some cases, the mode, median or mean value may not be
readily available through the corresponding attribute.
To paliate this, `ProbabilisticTensorDictModule` will first attempt
to get the value through a call to `get_mode()`, `get_median()` or `get_mean()`
if the method exists.
- **distribution_class** (*Type**or**Callable**[**[**Any**]**,**Distribution**]**,**optional*) -

keyword-only argument.
A `torch.distributions.Distribution` class to
be used for sampling.
Default is [`Delta`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta).

Note

If the distribution class is of type
[`CompositeDistribution`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution), the `out_keys`
can be inferred directly form the `"distribution_map"` or `"name_map"`
keyword arguments provided through this class' `distribution_kwargs`
keyword argument, making the `out_keys` optional in such cases.
- **distribution_kwargs** (*dict**,**optional*) -

keyword-only argument.
Keyword-argument pairs to be passed to the distribution.

Note

if your kwargs contain tensors that you would like to transfer to device with the module, or
tensors that should see their dtype modified when calling module.to(dtype), you can wrap the kwargs
in a `TensorDictParams` to do this automatically.
- **return_log_prob** (*bool**,**optional*) - keyword-only argument.
If `True`, the log-probability of the
distribution sample will be written in the tensordict with the key
log_prob_key. Default is `False`.
- **log_prob_keys** (*List**[**NestedKey**]**,**optional*) -

keys where to write the log_prob if `return_log_prob=True`.
Defaults to '<sample_key_name>_log_prob', where <sample_key_name> is each of the `out_keys`.

Note

This is only available when `composite_lp_aggregate()` is set to `False`.
- **log_prob_key** (*NestedKey**,**optional*) -

key where to write the log_prob if `return_log_prob=True`.
Defaults to 'sample_log_prob' when `composite_lp_aggregate()` is set to True
or '<sample_key_name>_log_prob' otherwise.

Note

When there is more than one sample, this is only available when `composite_lp_aggregate()` is set to `True`.
- **cache_dist** (*bool**,**optional*) - keyword-only argument.
EXPERIMENTAL: if `True`, the parameters of the
distribution (i.e. the output of the module) will be written to the
tensordict along with the sample. Those parameters can be used to re-compute
the original distribution later on (e.g. to compute the divergence between
the distribution used to sample the action and the updated distribution in
PPO). Default is `False`.
- **n_empirical_estimate** (*int**,**optional*) - keyword-only argument.
Number of samples to compute the empirical
mean when it is not available. Defaults to 1000.
- **generator** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**int**,**NestedKey**, or**None**,**optional*) - keyword-only argument.
Routes sampling through an explicit RNG instead of the global PyTorch RNG.
Accepts a [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) (used in place, advances across calls),
an `int` (shorthand for `Generator().manual_seed(int)`), or a
`NestedKey` to fetch the generator from the input tensordict on every
call (the value can be a `Generator` or a scalar int / Tensor used as a
JAX-style stream-key with a `next_seed` written back). Defaults to `None`,
in which case the global RNG is used. See
[`ProbabilisticTensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) for details.

Warning

Running checks takes time! Using safe=True will guarantee that the samples are within the spec bounds
given some heuristic coded in [`project()`](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.project), but that requires checking whether the
values are within the spec space, which will induce some overhead.

See also

:class`The composite distribution in tensordict <~tensordict.nn.CompositeDistribution>` can be used
to create multi-head policies.

Example

```
>>> from torchrl.modules import SafeProbabilisticModule
>>> from torchrl.data import Bounded
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import InteractionType
>>> mod = SafeProbabilisticModule(
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=torch.distributions.Normal,
... safe=True,
... spec=Bounded(low=-1, high=1, shape=()),
... default_interaction_type=InteractionType.RANDOM
... )
>>> _ = torch.manual_seed(0)
>>> data = TensorDict(
... loc=torch.zeros(10, requires_grad=True),
... scale=torch.full((10,), 10.0),
... batch_size=(10,))
>>> data = mod(data)
>>> print(data["action"]) # All actions are within bound
tensor([ 1., -1., -1., 1., -1., -1., 1., 1., -1., -1.],
 grad_fn=<ClampBackward0>)
>>> data["action"].mean().backward()
>>> print(data["loc"].grad) # clamp annihilates gradients
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

random(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/probabilistic.html#SafeProbabilisticModule.random)

Samples a random element in the target space, irrespective of any input.

If multiple output keys are present, only the first will be written in the input `tensordict`.

Parameters:

**tensordict** (*TensorDictBase*) - tensordict where the output value should be written.

Returns:

the original tensordict with a new/updated value for the output key.

random_sample(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/probabilistic.html#SafeProbabilisticModule.random_sample)

See `SafeModule.random(...)`.