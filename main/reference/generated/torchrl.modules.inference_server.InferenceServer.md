# InferenceServer

*class*torchrl.modules.inference_server.InferenceServer(*model: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *transport: [InferenceTransport](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)*, ***, *max_batch_size: int | None = None*, *min_batch_size: int | None = None*, *timeout: float | None = None*, *collate_fn: Callable | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *policy_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *output_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *collect_stats: bool | None = None*, *stats_window_size: int | None = None*, *weight_sync=None*, *weight_sync_model_id: str = 'policy'*, *server_config: [InferenceServerConfig](torchrl.modules.inference_server.InferenceServerConfig.html#torchrl.modules.inference_server.InferenceServerConfig) | None = None*, *device_config: [InferenceDeviceConfig](torchrl.modules.inference_server.InferenceDeviceConfig.html#torchrl.modules.inference_server.InferenceDeviceConfig) | None = None*, *shutdown_event: Event | Event | None = None*, *policy_version: int = 0*, *policy_version_key: NestedKey | None = 'policy_version'*)[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer)

Auto-batching inference server.

Actors submit individual TensorDicts via the *transport* and receive
results asynchronously. A background worker drains the transport queue,
batches inputs, runs the model, and fans results back to the callers.

Parameters:

- **model** (*nn.Module**or**Callable*) - a callable that maps a batched
TensorDictBase to a batched TensorDictBase (e.g. a
[`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)).
- **transport** ([*InferenceTransport*](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)) - the communication backend.

Keyword Arguments:

- **max_batch_size** (*int**,**optional*) - upper bound on the number of requests
processed in a single forward pass. Default: `64`.
- **min_batch_size** (*int**,**optional*) - minimum number of requests to
accumulate before dispatching a batch. After the first request
arrives the server keeps draining for up to `timeout` seconds
until at least this many items are collected. `1` (default)
dispatches immediately.
- **timeout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - seconds to wait for new work before
dispatching a partial batch. Default: `0.01`.
- **collate_fn** (*Callable**,**optional*) - function used to stack a list of
TensorDicts into a batch. Default: `lazy_stack()`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device to move batches to
before calling the model. This is kept as an alias for
`policy_device` for backward compatibility. `None` means no
device transfer.
- **policy_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device that owns the
policy and receives batched requests before model execution.
If omitted, `device` is used.
- **output_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device where individual
inference results are moved before being returned to actors. This
is useful when a CUDA policy serves CPU environment workers.
- **collect_stats** (*bool**,**optional*) - if `True`, collect lightweight
batching, queue-wait, and forward-latency statistics. Defaults to
`True`.
- **stats_window_size** (*int**,**optional*) - number of recent timing samples
kept for percentile statistics. Defaults to `1024`.
- **weight_sync** - an optional
[`WeightSyncScheme`](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme) used to receive
updated model weights from a trainer. When set, the server polls
for new weights between inference batches.
- **weight_sync_model_id** (*str**,**optional*) - the model identifier used when
initialising the weight sync scheme on the receiver side.
Default: `"policy"`.
- **server_config** ([*InferenceServerConfig*](torchrl.modules.inference_server.InferenceServerConfig.html#torchrl.modules.inference_server.InferenceServerConfig)*,**optional*) - structured server
configuration. Mutually exclusive with the `max_batch_size`,
`min_batch_size`, `timeout`, `collect_stats`, and
`stats_window_size` keyword arguments (passing any of them
alongside a config raises, even when the value equals the
default).
- **device_config** ([*InferenceDeviceConfig*](torchrl.modules.inference_server.InferenceDeviceConfig.html#torchrl.modules.inference_server.InferenceDeviceConfig)*,**optional*) - structured device
placement configuration. Mutually exclusive with `device`,
`policy_device`, and `output_device`. The server consumes
`policy_device` and `output_device` only; `env_device` is
used as a fallback for `output_device` and `storing_device`
is rejected (it is a collector-level setting).
- **policy_version** (*int**,**optional*) - initial behavior-policy version
attached to inference outputs. Defaults to `0`.
- **policy_version_key** (*NestedKey**or**None**,**optional*) - TensorDict key used
for behavior-policy version annotations. `None` disables
annotations. Defaults to `"policy_version"`.

Example

```
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import (
... InferenceServer,
... ThreadingTransport,
... )
>>> import torch.nn as nn
>>> policy = TensorDictModule(
... nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"]
... )
>>> transport = ThreadingTransport()
>>> server = InferenceServer(policy, transport, max_batch_size=8)
>>> server.start()
>>> client = transport.client()
>>> # client(td) can now be called from any thread
>>> server.shutdown()
```

client() → Any[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.client)

Return a restricted inference client from the owned transport.

*property*is_alive*: bool*

Whether the background worker thread is running.

*property*policy_version*: int*

The current behavior-policy version served with inference outputs.

shutdown(*timeout: float | None = 5.0*) → None[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.shutdown)

Signal the background worker to stop and wait for it to finish.

Parameters:

**timeout** (*float**or**None*) - seconds to wait for the worker thread to
join. `None` waits indefinitely.

start() → InferenceServer[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.start)

Start the background inference loop.

Returns:

self, for fluent chaining.

stats(***, *reset: bool = False*) → dict[str, float | int][[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.stats)

Return lightweight inference-server throughput statistics.

Parameters:

**reset** (*bool**,**optional*) - if `True`, clear counters after taking
the snapshot. Defaults to `False`.

Returns:

A dictionary with request/batch counts, rates, average batch size,
and p50/p95 queue and forward latencies in milliseconds.

update_model(*update_fn: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)], Any]*, ***, *mark_weight_update: bool = True*) → Any[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.update_model)

Apply an in-place update to the served model under the model lock.

Parameters:

- **update_fn** (*Callable*) - function called with `self.model` while
inference is blocked by the server's model lock.
- **mark_weight_update** (*bool**,**optional*) - if `True`, increment the
behavior-policy version and weight-update counter after
`update_fn` succeeds. Defaults to `True`.

Returns:

The value returned by `update_fn`.

update_policy_weights_(*model_id=None*, *policy_or_weights=None*, ***kwargs*)[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.update_policy_weights_)

Weight-sync cascade hook: record an applied weight update.

Weight-sync schemes cascade to their `context` after applying
weights to the registered model. The server installs itself as the
scheme context (when none is set) so that the policy version is
bumped exactly when weights are actually applied - including
shared-memory schemes whose background receiver thread applies
weights outside the server's polling loop.