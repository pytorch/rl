# InferenceServer

*class*torchrl.modules.inference_server.InferenceServer(*model: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *transport: [InferenceTransport](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)*, ***, *max_batch_size: int = 64*, *min_batch_size: int = 1*, *timeout: float = 0.01*, *collate_fn: Callable | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *weight_sync=None*, *weight_sync_model_id: str = 'policy'*)[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer)

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
before calling the model. `None` means no device transfer.
- **weight_sync** - an optional
[`WeightSyncScheme`](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme) used to receive
updated model weights from a trainer. When set, the server polls
for new weights between inference batches.
- **weight_sync_model_id** (*str**,**optional*) - the model identifier used when
initialising the weight sync scheme on the receiver side.
Default: `"policy"`.

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

*property*is_alive*: bool*

Whether the background worker thread is running.

shutdown(*timeout: float | None = 5.0*) → None[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.shutdown)

Signal the background worker to stop and wait for it to finish.

Parameters:

**timeout** (*float**or**None*) - seconds to wait for the worker thread to
join. `None` waits indefinitely.

start() → InferenceServer[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceServer.start)

Start the background inference loop.

Returns:

self, for fluent chaining.