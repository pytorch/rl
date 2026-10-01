# SharedMemWeightSyncScheme

*class*torchrl.weight_update.SharedMemWeightSyncScheme(*strategy: str = 'tensordict'*, *sync: bool = True*, *per_worker_weights: bool = False*)[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme)

Weight synchronization using shared memory.

This scheme uses shared memory for in-place weight updates. Workers
automatically see weight updates without explicit message passing.

A background thread on the receiver side listens for "receive" instructions
from the sender. When an instruction arrives, the thread applies the current
shared memory weights to the model and sends an acknowledgment.

Parameters:

- **strategy** - The weight transmission strategy (default: "tensordict").
- **sync** - If True (default), send() blocks until receiver acknowledges.
If False, send() returns immediately (use send_async/wait_async).
- **per_worker_weights** - If True, each worker maintains independent weights
(for distinct policy factories). If False (default), workers on the
same device share the same weight buffer. This flag is auto-set to
True when distinct policy factories are detected.

Example

```
>>> # Basic usage
>>> scheme = SharedMemWeightSyncScheme()
>>> # Weights are initialized via init_on_sender()
```

apply_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *inplace: bool = True*) → None

Apply weights to the model.

Parameters:

- **weights** - The weights to apply.
- **inplace** - Whether to apply weights in place. Default is True.

connect(***, *worker_idx: int | None = None*, *weights: Any | None = None*) → None

Method to be called once the workers have started.

Triggers a rendez-vous for the workers to receive their copy of the weights.

Dispatches to _setup_connection_and_weights_on_sender_impl() or _setup_connection_and_weights_on_receiver_impl()
based on which initialization was performed.

*property*context*: Any | None*

Get the context object (e.g., collector), if available.

Returns:

The context object if available, None otherwise.

create_transport(***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.create_transport)

Create shared memory transport.

Returns the shared transport instance that all workers will use.
Since this is shared memory, there's only one transport shared by all workers.

Note

This is used internally by init_on_sender/init_on_receiver.

*classmethod*from_backend(*backend: Literal['none', 'direct', 'shared', 'thread', 'process', 'multiprocessing', 'distributed', 'rpc', 'ray']*, ***kwargs*) → [WeightSyncScheme](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)

Construct a weight-sync scheme from its deployment backend.

Parameters:

- **backend** - One of `none`, `shared`, `process`, `distributed`,
`rpc`, or `ray`. `direct`, `thread`, and
`multiprocessing` are accepted aliases.
- ****kwargs** - Arguments forwarded to the concrete scheme constructor.

Returns:

The selected concrete scheme.

init_on_receiver(***, *model_id: str*, *context: Any = None*, ***kwargs*) → None

Initialize on worker process (receiver side).

This method is called once in each worker's initialization.

Parameters:

- **model_id** - Identifier for the model being synchronized
- **context** - Optional context object (e.g., inner collector)
- ****kwargs** - Alternative to context (model, etc.)

init_on_sender(**args*, ***kwargs*) → None

Initialize on the main process (sender side).

This method is called once in the collector's _run_processes() method,
after workers have been started and are ready to receive messages.

*property*model*: Any | None*

Get the model object, if available.

Returns:

The model object if available, None otherwise.

*property*model_id*: str | None*

Get the model ID for this scheme.

Returns:

The model ID if set, None otherwise.

prepare_weights(*weights: Any*, *model_id: str*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)*, *context: Any = None*) → Any[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.prepare_weights)

Prepare weights for SharedMemWeightSyncScheme.

When weights=None, we extract fresh weights from the model and update
the shared memory buffer in-place so workers see the change.

Parameters:

- **weights** - Raw weights input
- **model_id** - The model identifier
- **strategy** - WeightStrategy for extracting/converting weights
- **context** - Optional context (e.g., collector) for cache lookup

Returns:

Shared memory weights ready to send

receive(*timeout: float | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None

Check for and apply new weights (non-blocking).

This method is called in the worker's main loop to check if
new weights have been sent. If weights are available, they
are applied to the registered model immediately, and the update
is cascaded to any sub-collectors via context.update_policy_weights_().

Parameters:

**timeout** - Maximum time to wait for weights (seconds).
None means no timeout (blocking). Some transports may not
support timeout and will raise ValueError if specified.

Returns:

The received weights if available, None otherwise.

Note: For SharedMemWeightSyncScheme, this always returns None
since workers automatically see updates via shared memory.

*property*receiver_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the receiver transport.

Returns:

The receiver transport.

send(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *worker_ids: int | list[int] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.send)

Send weights via shared memory (in-place update).

For SharedMemWeightSyncScheme:
1. Updates the shared memory buffer(s) in-place
2. _send_instruction() tells workers to apply the new weights
3. If sync=True, waits for acknowledgments from all workers

Parameters:

- **weights** - Weights to send. Can be:
- None: Extract weights from the model
- TensorDictBase: Same weights broadcast to all workers
- dict[int, TensorDictBase]: Per-worker weights (keys are worker indices)
- **worker_ids** - Which workers to notify (None = all workers).

*property*sender_transports*: dict[int, [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)]*

Get the sender transports.

Returns:

The sender transports.

*property*shared_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the shared transport.

Returns:

The shared transport.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.shutdown)

Stop the background receiver thread and clean up.

*property*weights*: Any | None*

Get the current weights from shared memory.

For SharedMemWeightSyncScheme:
- On sender side: weights are in transport's _unique_weights
- On receiver side: weights are in _receiver_shared_weights (stored during connect())

Returns:

The weights TensorDict if available, None otherwise.

*property*worker_idx*: int | None*

Get the worker index for this scheme.

Returns:

The worker index if set, None otherwise.