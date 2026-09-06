# PolicyClientModule

*class*torchrl.modules.inference_server.PolicyClientModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/inference_server/_client.html#PolicyClientModule)

TensorDict policy wrapper for remote inference-server clients.

`PolicyClientModule` makes a transport client look like a TorchRL policy:
it accepts a [`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase), submits it to an
[`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer), and returns the
TensorDict produced by the remote policy. It can be passed anywhere a
TensorDict policy module is expected.

This class is the reference implementation of TorchRL's service *client*
contract: it duck-types the domain interface (a policy client IS a
TensorDict policy, so consumer code cannot tell local from remote), it is
cheap and picklable (it can be handed to spawned workers), and it carries
no lifecycle rights - clients can call the service but never start or
shut it down; only the owner that constructed the server can.

Note

Unlike a local [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule), the result
crosses a transport boundary, so `forward()` returns a *new*
TensorDict rather than writing the `out_keys` into the input
TensorDict. Use the return value; do not rely on in-place updates of
the input.

Parameters:

**client** (*Callable**or*[*InferenceTransport*](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)) - actor-side inference client.
If a transport is provided, `transport.client()` is called.

Keyword Arguments:

- **in_keys** (*sequence**of**NestedKey**,**optional*) - input keys advertised by the
module. The full input TensorDict is still sent to the server.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - output keys advertised by
the module.
- **max_inflight** (*int**,**optional*) - maximum number of unresolved
asynchronous requests submitted through this module; further
`submit()` calls block until a slot frees up. A slot is
freed when its request *completes* (including errors), not when
`result()` is first called; a timed-out `result()` keeps the
slot. Must be at least `1`. `None` means unbounded.

Note

The caller's active `tensordict.nn.interaction_type()` is
automatically attached to every transport request, and the server
executes the remote policy under that exploration context - exactly
as a local policy would see it. In-process (plain callable) clients
need no propagation since the caller's context is already active.

Note

Version tracking is an instance of the generic *service-stamped
metadata* pattern: a service may stamp every response with metadata
describing the state it was served from (here: the behavior-policy
version), and the data pipeline may enforce freshness constraints on
that metadata. Bounded-staleness enforcement lives in the replay
buffer through [`PolicyAgeFilter`](torchrl.envs.transforms.PolicyAgeFilter.html#torchrl.envs.transforms.PolicyAgeFilter),
which silently drops too-old elements on extend or sample instead of
raising in the consumer.

Note

The default `"policy_version"` key is shared on purpose with the
[`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion) transform and the
collectors' `track_policy_version` mechanism: they stamp the same
concept (the behavior-policy version that produced the data), so
consumers such as
[`PolicyAgeFilter`](torchrl.envs.transforms.PolicyAgeFilter.html#torchrl.envs.transforms.PolicyAgeFilter) can read it without
caring which component wrote it. Both counters are driven by the same
weight-update cascade (`update_policy_weights_`), so they agree when
wired through a weight-sync scheme. Keep a single authoritative writer
per data stream - in a policy-server topology that is the server,
which owns the weights; do not stack an independently-initialized
`PolicyVersion` transform on top of server-stamped data.

Examples

```
>>> import torch
>>> import torch.nn as nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import (
... InferenceServer,
... PolicyClientModule,
... ThreadingTransport,
... )
>>> policy = TensorDictModule(
... nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> transport = ThreadingTransport()
>>> server = InferenceServer(policy, transport).start()
>>> remote_policy = PolicyClientModule(
... transport, in_keys=["observation"], out_keys=["action"]
... )
>>> td = remote_policy(TensorDict({"observation": torch.randn(4)}))
>>> "action" in td.keys()
True
>>> server.shutdown()
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/inference_server/_client.html#PolicyClientModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

submit(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → Future | _ImmediateFuture[[source]](../../_modules/torchrl/modules/inference_server/_client.html#PolicyClientModule.submit)

Submit a TensorDict request and return a future-like object.

Parameters:

**tensordict** (*TensorDictBase*) - observation TensorDict to send to the
remote policy.

Returns:

Future-like object whose `result()` method returns a TensorDict.
When the wrapped client exposes `submit` this is the transport's
`Future` and submission errors raise
synchronously; for a plain callable client the call runs eagerly
and errors are deferred to `result()` on a reduced future that
only implements `done()` and `result()`.