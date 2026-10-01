# RandomSampler

*class*torchrl.data.replay_buffers.RandomSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#RandomSampler)

A uniformly random sampler for composable replay buffers.

Keyword Arguments:

**replacement** (*bool**,**optional*) - if `False`, the call is dispatched to
[`SamplerWithoutReplacement`](torchrl.data.replay_buffers.SamplerWithoutReplacement.html#torchrl.data.replay_buffers.SamplerWithoutReplacement), and any additional keyword
arguments (e.g. `drop_last`, `shuffle`) are forwarded to its
constructor. Defaults to `True`.

Examples

```
>>> from torchrl.data import RandomSampler, SamplerWithoutReplacement
>>> isinstance(RandomSampler(), RandomSampler)
True
>>> isinstance(RandomSampler(replacement=False), SamplerWithoutReplacement)
True
>>> isinstance(
... RandomSampler(replacement=False, drop_last=True),
... SamplerWithoutReplacement,
... )
True
```