# PolicyAgeFilter

*class*torchrl.envs.transforms.PolicyAgeFilter(*current_version: int | Callable[[], int]*, *max_policy_lag: int*, ***, *policy_version_key: NestedKey = 'policy_version'*, *strict: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/rb_transforms.html#PolicyAgeFilter)

Filter out data produced by a behavior policy that is too old.

Services such as [`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer)
stamp every response with the behavior-policy version that produced it
(the *service-stamped metadata* pattern). This transform enforces a
bounded-staleness constraint on that metadata inside the data pipeline:
elements whose stamped version lags the live version by more than
`max_policy_lag` weight updates are dropped, instead of raising in the
consumer.

Attached to a [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer), the transform filters
on both paths:

- on [`extend()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend) (inverse path), stale
elements never enter the buffer;
- on [`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample) (forward path), elements
that have become stale *since insertion* are dropped from the batch, so
the returned batch may be smaller than the requested batch size.

Attached to an environment, the transform is a no-op: data flowing
through an env pipeline is produced by the live policy and carries no
lag by construction.

Parameters:

- **current_version** (*int**or**Callable**[**[**]**,**int**]*) - live source of the
current policy version, e.g. `lambda: server.policy_version` or
`lambda: collector.policy_version`. A callable is re-evaluated
on every filtering pass; an `int` freezes the reference version.
- **max_policy_lag** (*int*) - maximum allowed
`current_version - stamped_version`.

Keyword Arguments:

- **policy_version_key** (*NestedKey**,**optional*) - key carrying the stamped
behavior-policy version. Must match the stamping service's
`policy_version_key`. Defaults to `"policy_version"`.
- **strict** (*bool**,**optional*) - if `True`, data without the version key
raises a `KeyError`; otherwise it passes through unfiltered
with a one-time warning. Defaults to `False`.

Note

Filtering produces data-dependent batch sizes, which is unfriendly to
`torch.compile`; keep the filter outside compiled regions.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyStackStorage, ReplayBuffer
>>> from torchrl.envs.transforms import PolicyAgeFilter
>>> current_version = 3
>>> rb = ReplayBuffer(
... storage=LazyStackStorage(100),
... transform=PolicyAgeFilter(lambda: current_version, max_policy_lag=1),
... )
>>> data = TensorDict(
... {"observation": torch.randn(4, 3), "policy_version": torch.tensor([0, 2, 2, 3])},
... batch_size=[4],
... )
>>> indices = rb.extend(data) # version 0 is filtered out on write
>>> len(rb)
3
>>> sample = rb.sample(3) # remaining data is fresh enough
>>> sample.batch_size[0]
3
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/rb_transforms.html#PolicyAgeFilter.forward)

Drop stale elements from a sampled batch (replay-buffer read path).