# MultiProcessWeightSyncScheme

*class*torchrl.weight_update.MultiProcessWeightSyncScheme(*strategy: str = 'tensordict'*, *sync: bool = True*)[[source]](../../_modules/torchrl/weight_update/_mp.html#MultiProcessWeightSyncScheme)

Weight synchronization for multiprocess operations using queues.

This scheme creates transports that communicate via multiprocessing queues.
Unlike the parent SharedMemWeightSyncScheme which uses shared memory for in-place
updates, this scheme sends actual weight copies through queues to workers.

A background thread on the receiver side listens for "receive" instructions
from the sender. When an instruction arrives, the thread receives the weights
from the weight queue and applies them to the model.

It follows the same two-phase pattern as SharedMemWeightSyncScheme:

1. **init_on_sender()**: Stores the recipe for creating device-specific weights
(model reference, devices, mapping functions) without creating actual copies
2. **synchronize_weights()**: Creates device-specific weight copies on-demand,
sends them sequentially to workers via queues, allowing garbage collection
between workers to minimize memory usage

This approach avoids holding multiple weight copies in memory simultaneously,
which is especially beneficial for large models with many workers.

Synchronization flow:
- **init_on_sender()**: Store configuration and register worker queues
- **synchronize_weights()**: Create and send initial weights on-demand
- **init_on_receiver()**: Create receiver that reads from queue
- **send()**: Extract and send weight updates, wait for acknowledgments

Parameters:

- **strategy** - The weight transmission strategy (default: "tensordict").
Can be "tensordict" or "state_dict".
- **sync** - If True (default), send() blocks until receiver acknowledges.
If False, send() returns immediately (use send_async/wait_async).

Example

```
>>> # Basic usage with collector
>>> scheme = MultiProcessWeightSyncScheme()
>>> collector = MultiSyncCollector(
... create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
... policy=policy,
... frames_per_batch=100,
... total_frames=1000,
... weight_sync_schemes={"policy": scheme},
... )
>>> # scheme.collect() is called automatically by collector
>>> # Weights are created on-demand and sent to workers efficiently
```

Note

The on-demand weight creation means that synchronize_weights() will be
slower than if weights were pre-computed, but memory usage is significantly
reduced, especially when workers use different devices or when the model
is large.

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

create_transport(***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/_mp.html#MultiProcessWeightSyncScheme.create_transport)

Create an MPTransport using the provided queue.

Note

This is used internally by init_on_sender/init_on_receiver.

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

prepare_weights(*weights: Any*, *model_id: str*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)*, *context: Any = None*) → Any

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

send(*weights: Any = None*, *worker_ids: int | list[int] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_mp.html#MultiProcessWeightSyncScheme.send)

Send weights synchronously to workers.

This method:
1. Prepares weights (extracts from model if weights=None)
2. Sends weights to the weight queue
3. Sends "receive" instruction to workers' background threads
4. If sync=True, waits for acknowledgments from those workers

Parameters:

- **weights** - Weights to send. Can be:
- None: Extract from model via context.get_model(model_id)
- nn.Module: Extract weights from module
- TensorDict: Use directly
- dict: Convert to TensorDict
- **worker_ids** - Which workers to send to:
- None: Send to all workers (default)
- int: Send to single worker
- list[int]: Send to specific workers

Note: If sync=True (default), this is a blocking call that ensures
specified workers are updated before returning.

*property*sender_transports*: dict[int, [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)]*

Get the sender transports.

Returns:

The sender transports.

*property*shared_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the shared transport.

Returns:

The shared transport.

shutdown() → None

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