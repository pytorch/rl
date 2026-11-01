# ProbabilisticActor

*class*torchrl.modules.tensordict_module.ProbabilisticActor(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#ProbabilisticActor)

General class for probabilistic actors in RL.

The ProbabilisticActor class comes with default values for the out_keys (["action"])
and if the spec is provided but not as a
Composite object, it will be
automatically translated into `spec = Composite(action=spec)`

Parameters:

- **module** (*nn.Module*) - a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) used to map the input to
the output parameter space.
- **in_keys** (*str**or**iterable**of**str**or**dict*) - key(s) that will be read from the
input TensorDict and used to build the distribution. Importantly, if it's an
iterable of string or a string, those keys must match the keywords used by
the distribution class of interest, e.g. `"loc"` and `"scale"` for
the Normal distribution and similar. If in_keys is a dictionary,, the keys
are the keys of the distribution and the values are the keys in the
tensordict that will get match to the corresponding distribution keys.
- **out_keys** (*str**or**iterable**of**str*) - keys where the sampled values will be
written. Importantly, if these keys are found in the input TensorDict, the
sampling step will be skipped.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - keyword-only argument containing the specs
of the output tensor. If the module outputs multiple output tensors,
spec characterize the space of the first output tensor.
- **safe** (*bool*) - keyword-only argument. if `True`, the value of the output is checked against the
input spec. Out-of-domain sampling can
occur because of exploration policies or numerical under/overflow
issues. If this value is out of bounds, it is projected back onto the
desired space using the `TensorSpec.project`
method. Default is `False`.
- **default_interaction_type** ([*tensordict.nn.InteractionType*](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)*,**optional*) -

keyword-only argument.
Default method to be used to retrieve
the output value. Should be one of: `InteractionType.MODE`, `InteractionType.DETERMINISTIC`,
`InteractionType.MEDIAN`, `InteractionType.MEAN` or
`InteractionType.RANDOM` (in which case the value is sampled
randomly from the distribution).
TorchRL's `ExplorationType` class is a proxy to `InteractionType`.
Defaults to `InteractionType.DETERMINISTIC`.

Note

When a sample is drawn, the `ProbabilisticActor` instance will
first look for the interaction mode dictated by the
`interaction_type()`
global function. If this returns None (its default value), then the
default_interaction_type of the ProbabilisticTDModule
instance will be used. Note that
[`BaseCollector`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)
instances will use set_interaction_type to
`tensordict.nn.InteractionType.RANDOM` by default.
- **distribution_class** (*Type**,**optional*) -

keyword-only argument.
A `torch.distributions.Distribution` class to
be used for sampling.
Default is [`tensordict.nn.distributions.Delta`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta).

Note

if `distribution_class` is of type [`CompositeDistribution`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution),
the keys will be inferred from the `distribution_map` / `name_map` keyword arguments of that
distribution. If this distribution is used with another constructor (e.g., partial or lambda function)
then the out_keys will need to be provided explicitly.
Note also that actions will **not** be prefixed with an `"action"` key, see the example below
on how this can be achieved with a `ProbabilisticActor`.
- **distribution_kwargs** (*dict**,**optional*) - keyword-only argument.
Keyword-argument pairs to be passed to the distribution.
- **return_log_prob** (*bool**,**optional*) - keyword-only argument.
If `True`, the log-probability of the
distribution sample will be written in the tensordict with the key
'sample_log_prob'. Default is `False`.
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
in which case the global RNG is used. Useful when the agent's RNG stream must
be isolated from the environment's -- see Patterson et al.,
"Empirical Design in Reinforcement Learning" ([arXiv:2304.01315](https://arxiv.org/abs/2304.01315)).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.data import Bounded
>>> from torchrl.modules import ProbabilisticActor, NormalParamExtractor, TanhNormal
>>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
>>> action_spec = Bounded(shape=torch.Size([4]),
... low=-1, high=1)
>>> module = nn.Sequential(torch.nn.Linear(4, 8), NormalParamExtractor())
>>> tensordict_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
>>> td_module = ProbabilisticActor(
... module=tensordict_module,
... spec=action_spec,
... in_keys=["loc", "scale"],
... distribution_class=TanhNormal,
... )
>>> td = td_module(td)
>>> td
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

Probabilistic actors also support compound actions through the
`tensordict.nn.CompositeDistribution` class. This distribution takes
a tensordict as input (typically "params") and reads it as a whole: the
content of this tensordict is the input to the distributions contained in the
compound one.

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution, TensorDictModule
>>> from torchrl.modules import ProbabilisticActor
>>> from torch import nn, distributions as d
>>> import torch
>>>
>>> class Module(nn.Module):
... def forward(self, x):
... return x[..., :3], x[..., 3:6], x[..., 6:]
>>> module = TensorDictModule(Module(),
... in_keys=["x"],
... out_keys=[("params", "normal", "loc"),
... ("params", "normal", "scale"),
... ("params", "categ", "logits")])
>>> actor = ProbabilisticActor(module,
... in_keys=["params"],
... distribution_class=CompositeDistribution,
... distribution_kwargs={"distribution_map": {
... "normal": d.Normal, "categ": d.Categorical}}
... )
>>> data = TensorDict({"x": torch.rand(10)}, [])
>>> actor(data)
TensorDict(
 fields={
 categ: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 normal: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 params: TensorDict(
 fields={
 categ: TensorDict(
 fields={
 logits: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 normal: TensorDict(
 fields={
 loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 x: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Using a probabilistic actor with a composite distribution can be achieved using the following
example code:

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> from tensordict.nn import TensorDictModule
>>> from torch import distributions as d
>>> from torch import nn
>>>
>>> from torchrl.modules import ProbabilisticActor
>>>
>>>
>>> class Module(nn.Module):
... def forward(self, x):
... return x[..., :3], x[..., 3:6], x[..., 6:]
...
>>>
>>> module = TensorDictModule(Module(),
... in_keys=["x"],
... out_keys=[
... ("params", "normal", "loc"), ("params", "normal", "scale"), ("params", "categ", "logits")
... ])
>>> actor = ProbabilisticActor(module,
... in_keys=["params"],
... distribution_class=CompositeDistribution,
... distribution_kwargs={"distribution_map": {"normal": d.Normal, "categ": d.Categorical},
... "name_map": {"normal": ("action", "normal"),
... "categ": ("action", "categ")}}
... )
>>> print(actor.out_keys)
[('params', 'normal', 'loc'), ('params', 'normal', 'scale'), ('params', 'categ', 'logits'), ('action', 'normal'), ('action', 'categ')]
>>>
>>> data = TensorDict({"x": torch.rand(10)}, [])
>>> module(data)
>>> print(actor(data))
TensorDict(
 fields={
 action: TensorDict(
 fields={
 categ: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 normal: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 params: TensorDict(
 fields={
 categ: TensorDict(
 fields={
 logits: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 normal: TensorDict(
 fields={
 loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 x: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```